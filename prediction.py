from utilities import cleanup, detokenize, parallel_process
from tqdm.auto import tqdm
import preprocessor as p
import nltk
import numpy as np

def prepare_input_category(df, targets):
    df_test = df.copy()
    types = targets

    def format_multiclass(item):
        return item.split("/")

    def format_multiclass_2(item):
        if item != item:
            return []
        return item.upper().split(",")

    def format_class(item):
        for ele in item:
            if ele.strip() in types:
                continue
            else:
                item.remove(ele)
        return item

    def format_tokenize(item):
        temp = ''
        for ele in item:
            temp = temp + '<' + ele.strip() + '>'
        return temp
    
    df_test["Label"] = df_test["Target"]
    df_test["Label_2"] = df_test["Target_2"]
    df_test["Label"] = np.where(
        df_test["Label"] == "Islamophobia", "MUSLIMS", df_test["Label"])
    df_test["Label"] = df_test["Label"].map(format_multiclass)
    df_test["Label_2"] = df_test["Label_2"].map(format_multiclass_2)
    df_test["Label"] = df_test["Label"] + df_test["Label_2"]

    df_test["Label"] = df_test["Label"].map(format_class)
    df_test["Label"] = df_test["Label"].map(format_class)
    df_test["Label"] = df_test["Label"].map(format_tokenize)

    df_test["input"] = '<|endoftext|>' + df_test["Label"] + "Hate-speech: " + \
        df_test['Hate_Speech'] + " " + "Counter-speech: "

    return df_test


def prepare_input(df, model_type):
    # load input sentences from the dataframe,
    # choose data format based on given model architecture

    # GPT
    if model_type == "GPT":
        # concatenate hate-speech and counter-speech
        df["input"] = '<|endoftext|>' + "Hate-speech: " + \
            df['Hate_Speech'] + " Counter-speech: "

    # BART
    elif model_type == "BART":
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
    if model_type == "GPT2":
        df = parallel_process(df, remove_prompt)

    # BART
    elif model_type == "BART":
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
    nltk.download('punkt')
    df["Prediction"] = df['Prediction'].map(remove_incomplete)
    cleanup()
    return df


def clean_text(item):
    # Text cleaning function, removes URLs, EMOJIs and extra whitespaces or special characters
    item = p.tokenize(detokenize(item))
    return item


def remove_prompt(df):
    # Predictions from GPT models contain the original prompts at the beginning,
    # remove the original prompt from generated contents to obtain the actual predictions
    cleaned_output = []
    for index, row in df.iterrows():
        prompt = row["input"].replace("<|endoftext|>", "")
        pred = row["Prediction"]
        pred = pred.replace(prompt, "")
        cleaned_output.append(pred)
    df["Prediction"] = cleaned_output
    return df


def remove_incomplete(text):
    # remove incomplete sentences at the end using nltk sentencizing
    # Split input string into individual sentences,
    # then remove any sentence that doesn't end with a punctuation mark.
    sentences = nltk.sent_tokenize(text)
    completed_sentences = [
        sentence for sentence in sentences if sentence[-1] in ['.', '!', '?', ')', ']', '$']]
    output = ' '.join(completed_sentences)
    return output
