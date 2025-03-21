import os
import preprocessor as p
import nltk
from tqdm.auto import tqdm

from lm_against_hate.utilities.misc import get_datetime, detokenize
from lm_against_hate.utilities.cleanup import cleanup_resources


def predict(ds, model, tokenizer, batchsize=64, max_gen_len=128, model_type=None, num_beams=3, no_repeat_ngram_size=3, num_return_sequences=1):
    outputs = []

    add_special_tokens = False if tokenizer.chat_template is not None else True

    for i in tqdm(range(0, len(ds), batchsize), desc="Predicting..."):
        cleanup_resources()
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


def ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
            
def post_processing(df, model_type:str, chat_template):
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
    cleanup_resources()

    ensure_nltk_resources()

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
    cleanup_resources()
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
