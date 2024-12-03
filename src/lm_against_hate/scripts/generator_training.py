import os

from transformers import Trainer, EarlyStoppingCallback
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json

from ..config.config import optuna_hp_space, categories, device
from ..utilities.hyperparameter_tuning import hyper_param_search
from ..utilities.data_util import dataloader_init
from ..utilities.model_loader import load_model, model_selection, save_trained_model, add_category_tokens
from ..utilities.misc import print_gpu_utilization


json_file_path = "./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
    print('loading huggingface credentials: ', credentials)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_TOKEN"] = credentials["HF_TOKEN"]


def main(modeltype: str, modelname: str = None, category: bool = False, save_option: bool = True, from_checkpoint: bool = False):
    '''
    Decide what model to train

    modelname: model name from huggingface model hub. Set to None if using GPT-2 medium or BART-large
    modeltype: Seq2Seq like BART, Causal like GPT
    '''
    print_gpu_utilization()

    # Load Model Parameters
    print(f'Loading model parameters for: {modeltype} {modelname}')
    params = model_selection(model_type=modeltype, model_name=modelname)

    # Load Category Embedding Option
    params['category'] = category
    print('Catagory Embedding Option: ', params['category'])

    print('This session will run on: ', device)
    
    # Load pre-trained tokenizer and model
    print('Loading base model: ', params['model_name'])
    tokenizer, model = load_model(
        model_type=modeltype, params=params)

    # Add target demographic token to tokenizer as required
    if params['category']:
        print('Adding the following categories to the tokenizer: ', categories)
        tokenizer, model = add_category_tokens(categories, tokenizer, model)

    model.config.use_cache = False
    model.print_trainable_parameters()

    # Load DataLoader and tokenize data
    print('Loading data')
    dataloader = dataloader_init(
        param=params, tokenizer=tokenizer, model=model, model_type=modeltype)
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)
    

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
        train_dataset=dataloader.ds['train'],
        eval_dataset=dataloader.ds['val'],
        data_collator=dataloader.datacollator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    trainer.train(resume_from_checkpoint=from_checkpoint)
    save_trained_model(tokenizer, model, params, save_option=save_option,
               targetawareness=params['category'])


if __name__ == "__main__":
    main(modeltype='S2S',                       # Causal for GPT, S2S for BART
         # specify base model if wanted, default is set in param
         modelname='google/flan-t5-xl',
         category=False,
         from_checkpoint=False)

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
