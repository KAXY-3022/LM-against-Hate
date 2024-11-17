import sys
sys.path.append('.')
import os
import json
import warnings
json_file_path = "./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
    print('loading huggingface credentials: ', credentials)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_TOKEN"] = credentials["HF_TOKEN"]
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from accelerate import Accelerator
from transformers import Trainer, EarlyStoppingCallback
from utilities.hyperparameter_tuning import hyper_param_search
from utilities.utils import save_model, model_selection, load_model, print_gpu_utilization
from utilities.data_util import data_init, add_category_tokens
from param import optuna_hp_space, categories

def main(modeltype: str, modelname: str = None, category: bool = False):
    '''
    Decide what model to train

    modelname: model name from huggingface model hub. Set to None if using GPT-2 medium or BART-large
    modeltype: Seq2Seq like BART, Causal like GPT
    '''
    print_gpu_utilization()
    
    # Load Model Parameters
    print('Loading model parameters for: ', modeltype, '', modelname)
    params = model_selection(modeltype=modeltype, modelname=modelname)

    # Load Category Embedding Option
    params['category'] = category
    print('Catagory Embedding Option: ', params['category'])
    
    # Check Device Availability
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    print('This session will run on: ', device)


    '''
    initiate pretrained tokenizer and model
    '''
    print('Loading base model: ', params['model_name'])
    tokenizer, model = load_model(modeltype=modeltype, params=params, device=device)

    # Add each target demographic as a new token to tokenizer if selected
    if params['category']:
        print('Adding the following categories to the tokenizer: ', categories)
        tokenizer, model = add_category_tokens(categories, tokenizer, model)

    model.config.use_cache = False
    model.print_trainable_parameters()
    
    
    # DATA PREPROCESSING and Batching
    print('Batching data')
    train_dataset, val_dataset, data_collator = data_init(modeltype=modeltype, 
                                                          params=params, 
                                                          tokenizer=tokenizer,
                                                          model=model,
                                                          category=params['category'])
    
    
    '''
    Hyperparameter Tuning
    '''
    '''
    best_trial = hyper_param_search(
    params['training_args'],
    optuna_hp_space,
    lm_dataset['train'],
    lm_dataset['test'],
    data_collator)

    setattr(params['training_args'], 'learning_rate', best_p['learning_rate'])
    setattr(params['training_args'], 'weight_decay', best_p['weight_decay'])
    setattr(params['training_args'], 'warmup_ratio', best_p['warmup_ratio'])
    '''

    '''
    TRAINING
    '''
    print('Training Starts')
    trainer = Trainer(
        model=model,
        args=params['training_args'],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    trainer.train()
    save_model(tokenizer, model, params, save_option=True, targetawareness=params['category'])


if __name__ == "__main__":
    main(modeltype='Causal',                       # Causal for GPT, S2S for BART
         # specify base model if wanted, default is set in param
         modelname='meta-llama/Llama-3.2-3B-Instruct',
         category=True)

    # openai-community/gpt2-medium          Causal
    # openai-community/gpt2-xl              Causal
    # facebook/bart-large                   Seq2Seq
    # google/flan-t5-large                  Seq2Seq
    # google/flan-t5-xl                     Seq2Seq
    # google/flan-t5-xxl                    Seq2Seq
    # meta-llama/Llama-3.2-1B-Instruct      Causal
    # meta-llama/Llama-3.2-3B-Instruct      Causal
    # meta-llama/Llama-3.1-8B-Instruct      Causal
    # mistralai/Mistral-7B-Instruct-v0.3    Causal
