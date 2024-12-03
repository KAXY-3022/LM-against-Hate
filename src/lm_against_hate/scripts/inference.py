import torch
from pathlib import Path

from ..utilities.data_util import dataloader_init, load_custom_dataset
from ..utilities.cleanup import cleanup_resources
from ..utilities.model_loader import load_model
from ..inference.inf_util import predict, post_processing, save_prediction
from ..config.inf_config import INFERENCE_MODELS


def main(models, datasets):
    # set data paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Predict
    for m in models:
        model, tokenizer = load_model(model_type=m['model_type'], params=m,local_only=True)
        for dataset_name in datasets:
            dataloader = dataloader_init(param=m , tokenizer=tokenizer, model=model, model_type=m['model_type'])
            dataloader.eval()
            
            print('Inferencing for dataset ', dataset_name)
            load_custom_dataset(ds_name=dataset_name, dataloader=dataloader)
            dataloader.prepare_dataset(tokenizer=tokenizer)

            df["Prediction"] = predict(
                dataloader.ds[dataset_name],
                model,
                tokenizer, 
                batchsize=64, 
                max_gen_len=128,
                model_type=m["model_type"], 
                num_beams=3, 
                no_repeat_ngram_size=3, 
                num_return_sequences=1
                )
            cleanup_resources()
        
            # Post Processing
            df = post_processing(df=df,
                                model_type=m["model_type"],
                                chat_template=tokenizer.chat_template)
            cleanup_resources() 

            save_dir = Path().resolve().joinpath('predictions', dataset_name)
            # Save predictions
            save_prediction(df, save_dir, m["model_name"], dataset_name)
            cleanup_resources()   




if __name__ == "__main__":
    main(
        # select one or multiple models to do inference
        models=INFERENCE_MODELS,
        datasets = ['Base',             # select one or multiple datasets to do inference
                    'Small', 
                    'Sexism',
                    ]
) 