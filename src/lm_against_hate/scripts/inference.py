from lm_against_hate.utilities.data_util import dataloader_init, load_custom_dataset
from lm_against_hate.utilities.cleanup import cleanup_resources
from lm_against_hate.utilities.model_loader import load_model, get_inference_model_list, MODEL_CONFIGS
from lm_against_hate.inference.inf_util import predict, post_processing, save_prediction
from lm_against_hate.config.config import pred_dir, model_path
from lm_against_hate.config.inf_config import INFERENCE_MODELS


def main(selected_only: bool, datasets: list):    
    # Predict
    if selected_only:
        models = INFERENCE_MODELS
    else:
        models = get_inference_model_list(model_path)
        
    for m in models:
        print('\nInferencing for model ', m['model_name'])
        model, tokenizer = load_model(model_type=m['model_type'], params=m, local_only=True)
        
        dataloader = dataloader_init(param=m , tokenizer=tokenizer, model=model, model_type=m['model_type'])
        dataloader.eval()
        dfs = {}
        for dataset_name in datasets:
            load_custom_dataset(ds_name=dataset_name, dataloader=dataloader)
            dfs[dataset_name] = dataloader.df[dataset_name]
            
        dataloader.prepare_dataset(tokenizer=tokenizer)
        
        for dataset_name in datasets:
            dfs[dataset_name]["Prediction"] = predict(
                dataloader.ds[dataset_name],
                model,
                tokenizer, 
                batchsize=8, 
                max_gen_len=128,
                model_type=m["model_type"], 
                num_beams=3, 
                no_repeat_ngram_size=3, 
                num_return_sequences=1
                )
            cleanup_resources()
        
            # Post Processing
            dfs[dataset_name] = post_processing(df=dfs[dataset_name],
                                model_type=m["model_type"],
                                chat_template=tokenizer.chat_template)
            cleanup_resources() 

            # Save predictions
            save_prediction(dfs[dataset_name], pred_dir.joinpath(
                dataset_name), m["model_name"], dataset_name)
            cleanup_resources()   




if __name__ == "__main__":
    main(
        # select one or multiple models to do inference
        selected_only = True,
        datasets = [#'Base',             # select one or multiple datasets to do inference
                    'Small', 
                    'Sexism',
                    ]
) 