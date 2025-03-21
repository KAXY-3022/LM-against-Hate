import torch._dynamo
import gc
import logging
import os
# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import json
import torch
from torch.utils.data import DataLoader
from typing import Optional
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import (
    Trainer,
    EarlyStoppingCallback,
    Seq2SeqTrainer
)
from transformers import TrainerCallback
from accelerate import infer_auto_device_map

from lm_against_hate.config.config import optuna_hp_space, categories, device
from lm_against_hate.utilities.hyperparameter_tuning import hyper_param_search
from lm_against_hate.utilities.data_util import dataloader_init
from lm_against_hate.utilities.model_loader import load_model, model_selection, save_trained_model, add_category_tokens
from lm_against_hate.utilities.misc import print_gpu_utilization
from lm_against_hate.utilities.cleanup import cleanup_resources

def main(
    modeltype: str,
    modelname: Optional[str] = None,
    category: bool = False,
    save_option: bool = True,
    from_checkpoint: bool = False,
    use_peft: bool = True,
    use_8bit: bool = True,
    use_flash_attention: bool = True
):
    """
    Train a language model with various optimizations.
    """
    torch.cuda.empty_cache()   
    device_map = infer_auto_device_map(
        model, max_memory={"cpu": "30GB", device: "12GB"}) if device == 'cuda:0' else device
    
    print_gpu_utilization()

    # Load Model Parameters
    print(f'Loading model parameters for: {modeltype} {modelname}')
    params = model_selection(model_type=modeltype, model_name=modelname)

    # Load Category Embedding Option
    params['category'] = category
    print('Category Embedding Option: ', params['category'])
    
    # Load credentials
    json_file_path = "./credentials.json"
    with open(json_file_path, "r") as f:
        credentials = json.load(f)
        print('Loading huggingface credentials')
        os.environ["HF_TOKEN"] = credentials["HF_TOKEN"]

    
    # Load pre-trained model with optimizations
    print('Loading base model: ', params['model_name'])
    model, tokenizer = load_model(
        model_type=modeltype,
        params=params,
        use_8bit=use_8bit,
        use_peft=use_peft,
        use_flash_attention=use_flash_attention,
        device_map=device_map
    )
    
    
    if use_peft:
        model.config.use_cache = False
        model.print_trainable_parameters()

    # Load data
    print('Loading data...')
    dataloader = dataloader_init(
        param=params,
        tokenizer=tokenizer,
        model=model,
        model_type=modeltype
    )
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)

    train_dataset = dataloader.ds['train']
    eval_dataset = dataloader.ds['val']
    data_collator = dataloader.datacollator
    

    # Initialize trainer with optimizations
    print('Training Starts')
    trainer_class = Seq2SeqTrainer if modeltype == 'S2S' else Trainer
    trainer = trainer_class(
        model=model,
        args=params['training_args'],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1,
                early_stopping_threshold=0.01
            ),
            # InputOutputLoggerCallback(log_frequency=50)
        ]
    )

    # Train model
    trainer.train(resume_from_checkpoint=from_checkpoint)
    
    # Save model
    save_trained_model(
        tokenizer,
        model,
        params,
        save_option=save_option,
        targetawareness=params['category']
    )
    
    # Cleanup
    cleanup_resources()

if __name__ == "__main__":
    
    main(
        modeltype='Causal',
        modelname='meta-llama/Llama-3.2-1B-Instruct',
        category=True,
        from_checkpoint=False,
        use_peft=True,
        use_8bit=True,
        use_flash_attention=True
    )

    # Model options:
    # openai-community/gpt2-medium          Causal
    # openai-community/gpt2-xl              Causal
    # facebook/bart-large                   Seq2Seq
    # google/flan-t5-large                  Seq2Seq
    # google/flan-t5-xl                     Seq2Seq
    # google/flan-t5-xxl                    Seq2Seq
    # meta-llama/Llama-3.2-1B-Instruct      Causal
    # meta-llama/Llama-3.2-1B               Causal
    # meta-llama/Llama-3.2-3B-Instruct      Causal
    # mistralai/Mistral-7B-Instruct-v0.3    Causal
