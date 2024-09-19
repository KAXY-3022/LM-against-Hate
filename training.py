import pandas as pd
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, Trainer, Seq2SeqTrainer
import torch

from param import optuna_hp_space, types
from data_util import preprocess_data_GPT, data_init, add_category_tokens
from utilities import save_model, model_selection, load_model
from hyperparameter_tuning import hyper_param_search


def main(modeltype: str, modelname=None):
  '''
  Decide what model to train

  Seq2Seq models like BART
  Causal LM like GPT
  '''

  print('Loading model parameters for: ', modeltype, '', modelname)
  params = model_selection(modeltype=modeltype, modelname=modelname)

  if torch.cuda.is_available(): 
    dev = "cuda:0" 
  else: 
    dev = "cpu" 
  device = torch.device(dev) 
  print('This session will run on: ', device)

  # Read csv file into dataframe
  print('Loading dataset: ', params['train_dir'])
  df = pd.read_csv(params['train_dir'])
  if modeltype == 'Causal':
    df = preprocess_data_GPT(df, types)


  # Train/Val split
  print('Spliting dataset')
  dataset = Dataset.from_pandas(df)
  # since we already have a seperate test set, we use the [test] split to run validation
  dataset = dataset.train_test_split(test_size=0.2) 


  '''
  initiate pretrained tokenizer and model
  '''

  print('Loading base model: ', params['model_name'])
  tokenizer = AutoTokenizer.from_pretrained(params['model_name']) 
  config = AutoConfig.from_pretrained(params['model_name'])
  model = load_model(modeltype=modeltype, params=params, config=config)

  # Add each target demographic as a new token to tokenizer if selected
  if params['category']:
    print('Adding the following categories to the tokenizer: ', types)
    tokenizer, model = add_category_tokens(types, tokenizer, model)


  # DATA PREPROCESSING and Batching
  print('Batching data')
  lm_dataset, data_collator = data_init(
    modeltype=modeltype,
    tokenizer=tokenizer,
    dataset=dataset,
    model=model)


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
      model=model.to(device),
      args=params['training_args'],
      train_dataset=lm_dataset["train"],
      eval_dataset=lm_dataset["test"],
      data_collator=data_collator,
  )
  setattr(trainer.args, 'num_train_epochs', 20)

  trainer.train()

  # remove saved checkpoints
  # !rm -rf {my_model_name}



  '''
  SAVE MODEL
  '''

  save_model(tokenizer, model, params['save_dir'], params['save_name'], save_option=True)

if __name__ == "__main__":
  main(modeltype = 'Causal',        # Causal for GPT, S2S for BART
       modelname = None)            # specify base model if wanted, default is set in param