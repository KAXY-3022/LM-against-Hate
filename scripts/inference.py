import sys
sys.path.append('.')  

import torch
from pathlib import Path

from utilities.utils import cleanup, load_local_model
from utilities.inf_util import predict, post_processing, save_prediction, load_dataset


def main(models, datasets):
    # set data paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Predict
    for m in models:
        tokenizer, model = load_local_model(params=m, device=device)
        for dataset in datasets:
            print('Inferencing for dataset ', dataset)
            df, ds = load_dataset(ds_name=dataset,
                                  model_type=m["model_type"],
                                  gen_tokenizer=tokenizer,
                                  target_awareness=m["TA"])

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
            df = post_processing(df=df,
                                model_type=m["model_type"],
                                chat_template=tokenizer.chat_template)
            cleanup() 

            save_dir = Path().resolve().joinpath('predictions', dataset)
            # Save predictions
            save_prediction(df, save_dir, m["model_name"], dataset)
            cleanup()   


gpt2_medium = {
    "model_type": "Causal",
    "TA": False,
    "model_name": "gpt2-medium_16,05,2023-21,09"
    }

gpt2_medium_category = {
    "model_type": "Causal",
    "TA": True,
    "model_name": "gpt2-medium-category_11,06,2023--00,31"
}

gpt2_xl = {
    "model_type": "Causal",
    "TA": False,
    "model_name": "gpt2-xl_07,11,2024--21,00"
}

bart_large = {
    "model_type": "S2S",
    "TA": False,
    "model_name": "bart-large_18,05,2023--19,18"
    }

Llama_32_1B = {
    "model_type": "Causal",
    "TA": False,
    "model_name": "Llama-3.2-1B-Instruct_16,11,2024--00,22"
}

Llama_32_1B_category = {
    "model_type": "Causal",
    "TA": True,
    "model_name": "Llama-3.2-1B-Instruct-category_17,11,2024--09,39"
}

Llama_32_3B = {
    "model_type": "Causal",
    "TA": False,
    "model_name": "Llama-3.2-3B-Instruct_26,11,2024--00,57"
}

Llama_32_3B_category = {
    "model_type": "Causal",
    "TA": True,
    "model_name": "Llama-3.2-3B-Instruct-category_22,11,2024--01,23"
}

if __name__ == "__main__":
    main(
        # select one or multiple models to do inference
        models=[Llama_32_1B, Llama_32_1B_category, Llama_32_3B, Llama_32_3B_category],
        datasets = ['Base',             # select one or multiple datasets to do inference
                    'Small', 
                    'Sexism',
                    ]
) 