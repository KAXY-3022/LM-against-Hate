import os
import gc
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from pathlib import Path

from param import GPT2_params, BART_params


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parallel_process(df, fn_to_execute, num_cores=4):
    # create a pool for multiprocessing
    pool = Pool(num_cores)

    # split your dataframe to execute on these pools
    splitted_df = np.array_split(df, num_cores)

    # execute in parallel:
    split_df_results = pool.map(fn_to_execute, splitted_df)

    # combine your results
    df = pd.concat(split_df_results)

    pool.close()
    pool.join()
    return df


def get_datetime(format):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt = now.strftime(format)
    return dt


def detokenize(x):
    return x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")  # noqa


def model_selection(modeltype: str, modelname: Optional[str] ):
    if modeltype == 'S2S':
        params = BART_params
    elif modeltype == 'Causal':
        params = GPT2_params

    if modelname:
        params['model_name'] = modelname
    
    return params

def load_model(modeltype: str, params: dict):
    # Load base model from 
    load_path = Path().resolve().joinpath(
        params['save_dir'],
        params['training_args'].output_dir,
        params['model_name'])
    print('Loading local model from: ', load_path)
    if modeltype == 'Causal':
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path) 
            config = AutoConfig.from_pretrained(load_path)
            model = AutoModelForCausalLM.from_pretrained(load_path, config=config)
        except:
            print('No local model found, downloading base model from hub')
            tokenizer = AutoTokenizer.from_pretrained(params['model_name']) 
            config = AutoConfig.from_pretrained(params['model_name'])
            model = AutoModelForCausalLM.from_pretrained(params['model_name'], config=config)
            # Save base model for future use
            save_path = Path().resolve().joinpath(
                params['save_dir'], 
                params['training_args'].output_dir,
                params['model_name'])
            print('Saving base model locally for future usage at: ', save_path)
            model.save_pretrained(save_path)
    elif modeltype == 'S2S':
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path) 
            config = AutoConfig.from_pretrained(load_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(load_path, config=config)
        except:
            print('No local model found, downloading base model from hub')
            tokenizer = AutoTokenizer.from_pretrained(params['model_name']) 
            config = AutoConfig.from_pretrained(params['model_name'])
            model = AutoModelForSeq2SeqLM.from_pretrained(params['model_name'], config=config)
            # Save base model for future use
            save_path = Path().resolve().joinpath(
                params['save_dir'], 
                params['training_args'].output_dir,
                params['model_name'])
            print('Saving base model locally for future usage at: ', save_path)
            model.save_pretrained(save_path)

    return tokenizer, model

def get_model_path(root_dir, load_model_name, version):
    # construct the path to a given model by the model name and it's version (date_time when created)
    load_dir = os.path.join(root_dir, "models/")
    load_dir = os.path.join(load_dir, load_model_name)
    load_dir = os.path.join(load_dir, version)
    return load_dir


def save_prediction(df, save_dir, model_name, version, dataset):
    # save inference predictions from a model to a local directory
    file_name = model_name + "_" + version + "_" + dataset + "_" + \
        get_datetime("%d,%m,%Y--%H,%M") + ".csv"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(save_dir + file_name)


def save_model(tokenizer, model, params, save_option=True):
    # save model and tokenizer to a local directory.
    save_path = Path().resolve().joinpath(
        params['save_dir'],
        params['training_args'].output_dir,
        params['model_name'] + '-' + get_datetime("%d,%m,%Y--%H,%M"))

    if save_option:
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("Model saved at: ", save_path)
    else:
        print("Save option is currently disabled. If you wish to keep the current model for future usage, please turn on the saving option by setting save_option=True")

