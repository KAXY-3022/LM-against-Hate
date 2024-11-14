from data_util import tokenize_labels
from utilities import cleanup, detokenize, parallel_process
from tqdm.auto import tqdm
import preprocessor as p
import nltk    
nltk.download('punkt')
nltk.download('punkt_tab')
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from utilities import get_datetime
import os

def prepare_input_category(df, model_type):
    # tokenize target demographics for prompting <CLASS-1><CLASS-2>
    df = tokenize_labels(df)
    
    # set up classifier model for generating target demographic when none available
    model_path = Path().resolve().joinpath(
        'models', 
        'Classifiers',
        'cardiffnlp-tweet-topic-21-multi-09,06,2023--21,45')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map='cuda')
    
    # set up class mapping
    class_mapping = model.config.id2label
       
    def transform_row(r):
        '''
        For each row, generate target demographic label for instances without existing labels.
        During inference, we use existing human labels. If no human label exists, we backtrack to generated target demographic label 
        '''
        # if the stored Target is empty list, generate target demographic label
        if len(r['Target']) == 0:
            labels = []
            tokens = tokenizer(r['Hate_Speech'],
                               return_tensors='pt').to('cuda')
            output = model(**tokens)
            scores = output[0][0].detach().cpu().numpy()
            scores = expit(scores)
            predictions = (scores >= 0.5) * 1
            for i in range(len(predictions)):
                if predictions[i]:
                    labels.append(class_mapping[i])
                    
            r['Label'] = labels
        else:
            r['Label'] = r['Target']
        return r
    df = df.apply(transform_row, axis=1)
    
    # Concatenate inputs for diffrent model tpyes
    if model_type == 'Causal':
        df["input"] = '<|endoftext|>' + df["Label"] + "Hate-speech: " + \
            df['Hate_Speech'] + " " + "Counter-speech: "
        
    elif model_type == 'S2S':
        df["input"] = df["Label"] + "Hate-speech: "

    return df


def prepare_input(df, model_type):
    # load input sentences from the dataframe,
    # choose data format based on given model architecture

    # GPT
    if model_type == "Causal":
        # concatenate hate-speech and counter-speech
        df["input"] = '<|endoftext|>' + "Hate-speech: " + \
            df['Hate_Speech'] + " Counter-speech: "

    # BART
    elif model_type == "S2S":
        df["input"] = df["Hate_Speech"]

    # Default
    else:
        raise Exception(
            "The given model is of unkonwn architecture. Please check your spelling or define new preprocessing function.")
    return df


def predict(ds, model, tokenizer, batchsize=40, max_gen_len=128, model_type=None, num_beams=3, no_repeat_ngram_size=3, num_return_sequences=1):
    outputs = []

    for i in tqdm(range(0, len(ds), batchsize)):
        cleanup()
        batch = ds["input"][i: i+batchsize]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)

        if model_type == "GPT":
            max_length = len(inputs['input_ids'][0]) + \
                max_gen_len  # let it generate longer
        else:
            max_length = max_gen_len

        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
            pad_token_id=tokenizer.eos_token_id,
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


def post_processing(df, model_type):
    # remove NaN in predictions if any exists
    df["Prediction"] = df["Prediction"].fillna('')

    # set options for cleaning text data (convert URLs and Emojis into special tokens $URL$, $EMJ$ )
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    
    # GPT
    if model_type == "Causal":
        def transform_row(r):
            '''
            Predictions from GPT models contain the original prompts at the beginning,
            remove the original prompt from generated contents to obtain the actual predictions
            '''
            prompt = r["input"].replace("<|endoftext|>", "")
            r["Prediction"] = r["Prediction"].replace(prompt, "")
            return r
        df = df.apply(transform_row, axis=1)

    # BART
    elif model_type == "S2S":
        pass

    # Default
    else:
        raise Exception(
            "The given model is of unkonwn architecture. Please check your spelling or define new post-processing function.")

    # Universal
    # drop the tokenized input coloum
    df = df.drop("input", axis=1)

    # clean and format the output, remove special characters and replacing emojis/URLs with $EMOJI$/$URL$
    df["Prediction"] = df['Prediction'].map(clean_text)
    cleanup()

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


def save_prediction(df, save_dir, model_name, version, dataset):
    # save inference predictions from a model to a local directory
    file_name = model_name + "_" + version + "_" + dataset + "_" + \
        get_datetime("%d,%m,%Y--%H,%M") + ".csv"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir.joinpath(file_name)
    print('saving predictions at ', save_path)

    df.to_csv(save_path)
