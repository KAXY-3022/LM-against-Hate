import os

import preprocessor as p
import nltk
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from datasets import Dataset
from scipy.special import expit
from tqdm.auto import tqdm

from utilities.utils import get_datetime, cleanup, detokenize, parallel_process
from utilities.data_util import tokenize_labels, prepare_input_causal_inf, prepare_input_noncausal
from param import data_path, model_path, system_message


def load_dataset(ds_name: str, model_type: str, gen_tokenizer, target_awareness: bool = False):
    '''
    ds_name: select which dataset to load
    model_type: S2S or Causal
    gen_tokenizer: tokenizer from the used generator, used to preprocess and tokenize input
    target_awareness: whether the model is trained with target demographic embeddings
    '''
    if ds_name == 'Base':
        # For base comparison
        file_name = 'CONAN_test.csv'

    elif ds_name == 'Small':
        # For comparison with ChatGPT
        file_name = 'T8-S10.csv'

    elif ds_name == 'Sexism':
        # For final comparison, sexism only test set
        file_name = 'EDOS_sexist.csv'

    load_path = data_path.joinpath(file_name)

    print('loading data from ', load_path)
    df_raw = pd.read_csv(load_path, converters={'Target': pd.eval})

    df, ds = prepare_input(
        df=df_raw,
        gen_tokenizer=gen_tokenizer,
        model_type=model_type,
        target_awareness=target_awareness)

    return df, ds


def prepare_input(df, gen_tokenizer, model_type: str, target_awareness: bool = False):
    if target_awareness:
        # set up classifier model for generating target demographic when none available
        load_path = model_path.joinpath(
            'Classifiers',
            'cardiffnlp-tweet-topic-21-multi_09,06,2023--21,45')
        class_tokenizer = AutoTokenizer.from_pretrained(load_path)
        if class_tokenizer.pad_token is None:
            class_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        class_model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            device_map='cuda')

        # set up class mapping
        class_mapping = class_model.config.id2label

        def generate_label(r):
            '''
            For each row, generate target demographic label for instances without existing labels.
            During inference, we use existing human labels. If no human label exists, we backtrack to generated target demographic label 
            '''
            # if the stored Target is empty list, generate target demographic label
            if len(r['Target']) == 0:
                labels = []
                tokens = class_tokenizer(r['Hate_Speech'],
                                         return_tensors='pt').to('cuda')
                output = class_model(**tokens)
                scores = output[0][0].detach().cpu().numpy()
                scores = expit(scores)
                predictions = (scores >= 0.5) * 1
                for i in range(len(predictions)):
                    if predictions[i]:
                        labels.append(class_mapping[i])

                r['Target'] = labels
            return r

        df = df.apply(generate_label, axis=1)
        
        # tokenize target demographics for prompting <CLASS-1><CLASS-2>
        df = tokenize_labels(df, column_name='Target')

    if gen_tokenizer.chat_template is not None:
        # if model is instruction tuned, preprocess input with chat_template
        df = prepare_input_noncausal(df=df, category=target_awareness)

        def format_chat(r):
            r['chat'] = [system_message,
                         {'role': 'user', 'content': r['text']}]
            return r

        df = df.apply(format_chat, axis=1)
        ds = Dataset.from_pandas(df)
        ds = ds.map(
            lambda x: {"formated_chat": gen_tokenizer.apply_chat_template(
                x["chat"], tokenize=False, add_generation_prompt=True)},
            desc="applying chat_template to dataset")

        def preprocess_function(examples):
            return gen_tokenizer(examples["formated_chat"], truncation=True, add_special_tokens=False)
        ds = ds.map(
            preprocess_function,
            batched=True,
            desc="tokenzing dataset")

    # Concatenate inputs for diffrent model tpyes
    elif model_type == 'Causal':
        df = prepare_input_causal_inf(df=df, category=target_awareness)
        ds = Dataset.from_pandas(df)

    elif model_type == 'S2S':
        df = prepare_input_noncausal(df=df, category=target_awareness)
        ds = Dataset.from_pandas(df)

    # Default
    else:
        raise Exception(
            "The given model is of unkonwn architecture. Please check your spelling or define new preprocessing function.")
    return df, ds


def predict(ds, model, tokenizer, batchsize=64, max_gen_len=128, model_type=None, num_beams=3, no_repeat_ngram_size=3, num_return_sequences=1):
    outputs = []

    if tokenizer.chat_template is not None:
        add_special_tokens = False
    else:
        add_special_tokens = True

    for i in tqdm(range(0, len(ds), batchsize), desc="Predicting..."):
        cleanup()
        batch = ds["text"][i: i+batchsize]
        inputs = tokenizer(batch, return_tensors="pt",
                           padding=True, add_special_tokens=add_special_tokens)

        if model_type == "Causal":
            max_length = len(inputs['input_ids'][0]) + \
                max_gen_len  # let it generate longer

            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )
            results = tokenizer.batch_decode(
                output_sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        else:
            max_length = max_gen_len

            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )
            results = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True)
        outputs.extend(results)
    return outputs


def post_processing(df, model_type, chat_template):
    # remove NaN in predictions if any exists
    df["Prediction"] = df["Prediction"].fillna('')

    # set options for cleaning text data (convert URLs and Emojis into special tokens $URL$, $EMJ$ )
    p.set_options(p.OPT.URL, p.OPT.EMOJI)

    # Universal
    # drop the tokenized input coloum
    if 'input' in df.columns:
        df = df.drop("input", axis=1)
    if 'text' in df.columns:
        df = df.drop("text", axis=1)
    if 'chat' in df.columns:
        df = df.drop("chat", axis=1)

    # clean and format the output, remove special characters and replacing emojis/URLs with $EMOJI$/$URL$
    df["Prediction"] = df['Prediction'].map(clean_text)
    cleanup()

    nltk.download('punkt')
    nltk.download('punkt_tab')

    # remove incomplete sentences at the end using nltk sentencizing
    def remove_incomplete(text):
        # remove incomplete sentences at the end using nltk sentencizing
        # Split input string into individual sentences,
        # then remove any sentence that doesn't end with a punctuation mark.
        sentences = nltk.sent_tokenize(text)
        completed_sentences = [
            sentence for sentence in sentences if sentence[-1] in ['.', '!', '?', ')', ']', '$']]
        output = ' '.join(completed_sentences)
        return output
    df["Prediction"] = df['Prediction'].map(remove_incomplete)
    cleanup()
    return df


def clean_text(item):
    # Text cleaning function, removes URLs, EMOJIs and extra whitespaces or special characters
    item = p.tokenize(detokenize(item))
    return item


def save_prediction(df, save_dir, model_name, dataset):
    # save inference predictions from a model to a local directory
    file_name = model_name + "_" + dataset + "_" + \
        get_datetime("%d,%m,%Y--%H,%M") + ".csv"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir.joinpath(file_name)
    print('saving predictions at ', save_path)

    df.to_csv(save_path)
