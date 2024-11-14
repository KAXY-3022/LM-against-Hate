# set data paths
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from pathlib import Path

from utilities import cleanup
from inf_util import prepare_input, predict, post_processing, prepare_input_category, save_prediction
from param import categories


def main(models, datasets):
    # set data paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = Path().resolve().joinpath('data', 'Custom')

    CONAN_test_dir = data_dir.joinpath('CONAN_test.csv')        # For base comparison
    CONAN_test_small_dir = data_dir.joinpath('T8-S10.csv')      # For comparison with ChatGPT
    EDOS_test_dir = data_dir.joinpath('EDOS_sexist.csv')        # For final comparison, sexism only test set

    # load local models
    def load_model(model_path, model_type):
        if model_type == "Causal":
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device)
            except ValueError:
                # If no support for flash_attention_2, then use standard implementation
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map=device)
            tokenizer.padding_side = "left" 
        elif model_type == "S2S":
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device)
            except ValueError:
                # If no support for flash_attention_2, then use standard implementation
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, device_map=device)
        tokenizer.pad_token = tokenizer.eos_token # to avoid an error
        cleanup()

        return model, tokenizer

    # Read csv file into dataframe

    def load_dataset(dataset_name, model_type, target_awareness):
        if dataset_name == 'Base':
            df_raw = pd.read_csv(CONAN_test_dir)
        elif dataset_name == 'Small':
            df_raw = pd.read_csv(CONAN_test_small_dir)
        elif dataset_name == 'Sexism':
            df_raw = pd.read_csv(EDOS_test_dir)
            df_raw = df_raw.rename(columns={"text": "Hate_Speech"})
            df_raw["Target"] = df_raw["labels"].map(lambda x: "WOMEN" if x == "sexist" else "other")

        # TODO: use categoty function to prepare inputs
        if target_awareness:
            df = prepare_input_category(df_raw, categories)
        else:
            df = prepare_input(df_raw, model_type)
        ds = Dataset.from_pandas(df)

        return df, ds

    # Predict
    for m in models:
        model_path = Path().resolve().joinpath('models', 
                                               m['model_type'], 
                                               m['load_model_name'] + '-' + m['version'])
        model, tokenizer = load_model(model_path, m["model_type"])
        for dataset in datasets:
            print('Inferencing for dataset ', dataset)
            df, ds = load_dataset(dataset, m["model_type"], m["TA"])
            df["Prediction"] = predict(
                ds,
                model,
                tokenizer, 
                batchsize=16, 
                max_gen_len=128,
                model_type=m["model_type"], 
                num_beams=3, 
                no_repeat_ngram_size=3, 
                num_return_sequences=1
                )
        cleanup()
        # Post Processing
        df = post_processing(df, m["model_type"])
        cleanup() 

        save_dir = Path().resolve().joinpath('predictions', dataset)
        # Save predictions
        save_prediction(df, save_dir, m["load_model_name"], m["version"], dataset)
        cleanup()   


gpt2_small = {
    "model_type": "Causal",
    "TA": False,
    "load_model_name": "gpt2",
    "version": "11,03,2023-16,02"
    }

gpt2_medium = {
    "model_type": "Causal",
    "TA": False,
    "load_model_name": "gpt2-medium",
    "version": "16,05,2023-21,09"
    }

gpt2_xl = {
    "model_type": "Causal",
    "TA": False,
    "load_model_name": "gpt2-xl",
    "version": "07,11,2024--21,00"
}

gpt2_medium_category = {
    "model_type": "Causal",
    "TA": True,
    "load_model_name": "gpt2-medium-category",
    "version": "11,06,2023--00,31"
    }

bart_small = {
    "model_type": "S2S",
    "TA": False,
    "load_model_name": "bart",
    "version": "17,03,2023-22,54"
    }

bart_large = {
    "model_type": "S2S",
    "TA": False,
    "load_model_name": "bart-large",
    "version": "18,05,2023--19,18"
    }

if __name__ == "__main__":
    main(
        # select one or multiple models to do inference
        models=[gpt2_xl],
        datasets = ['Base',             # select one or multiple datasets to do inference
                    # 'Small', 
                    # 'Sexism',
                    ]
) 