import os
import gc
import torch
import numpy as np
import pandas as pd
from pynvml import *
from pathlib import Path
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, AutoPeftModelForCausalLM

from multiprocessing import Pool
from param import Causal_params, S2S_params


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


def model_selection(modeltype: str, modelname: Optional[str]):
    if modeltype == 'S2S':
        params = S2S_params
    elif modeltype == 'Causal':
        params = Causal_params
    else:
        raise ValueError('no parameters for given modeltype {modeltype} found')
    if modelname:
        params['model_name'] = modelname
        params['save_name'] = modelname.split('/')[-1] + '_againstHate'

    return params


def load_model(modeltype: str, params: dict, device: str = 'cuda'):
    # Load base model from
    model_name = params['model_name'].split('/')[-1]
    load_path = params['save_dir'].joinpath(model_name)

    print('Loading local model from: ', load_path)

    if modeltype == 'Causal':
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_path, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device)

        except ValueError:
            # If no support for flash_attention_2, then use standard implementation
            model = AutoModelForCausalLM.from_pretrained(
                params['model_name'], device_map=device)

        except OSError:
            print('No local model found, downloading model from hub')
            tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    params['model_name'],
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device)

            except ValueError:
                # If no support for flash_attention_2, then use standard implementation
                model = AutoModelForCausalLM.from_pretrained(
                    params['model_name'], device_map=device)

            # Save base model for future use
            print('Saving model locally for future usage at: ', load_path)
            model.save_pretrained(load_path)
            tokenizer.save_pretrained(load_path)

    elif modeltype == 'S2S':
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device)

        except ValueError:
            # If no support for flash_attention_2, then use standard implementation
            model = AutoModelForSeq2SeqLM.from_pretrained(
                params['model_name'], device_map=device)

        except OSError:
            print('No local model found, downloading model from hub')
            tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    params['model_name'],
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device)

            except ValueError:
                # If no support for flash_attention_2, then use standard implementation
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    params['model_name'], device_map=device)

            # Save base model for future use
            print('Saving model locally for future usage at: ', load_path)
            model.save_pretrained(load_path)
            tokenizer.save_pretrained(load_path)

    tokenizer, model = add_pad_token(tokenizer, model)
    model.enable_input_require_grads()
    model = get_peft_model(model, params['peft_config'])
    return tokenizer, model


def add_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def load_local_model(params: dict, device: str = 'cuda'):
    # Load base model from
    load_path = Path().resolve().joinpath(
        'models', params['model_type'], params['model_name'])

    print('Loading local model from: ', load_path)

    if params['model_type'] == 'Causal':
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_path, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device)

        except RuntimeError:
            model = AutoPeftModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device)

        except ValueError:
            # If no support for flash_attention_2, then use standard implementation
            model = AutoModelForCausalLM.from_pretrained(
                params['model_name'], device_map=device)

        except OSError:
            raise OSError('No local model found')

    elif params['model_type'] == 'S2S':
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                load_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device)

        except ValueError:
            # If no support for flash_attention_2, then use standard implementation
            model = AutoModelForSeq2SeqLM.from_pretrained(
                params['model_name'], device_map=device)

        except OSError:
            raise OSError('No local model found')

    return tokenizer, model


def save_model(tokenizer, model, params, save_option=True, targetawareness: bool = False):
    # save model and tokenizer to a local directory.
    model_name = params['model_name'].split('/')[-1]
    if targetawareness:
        save_name = model_name + '-category' + \
            '_' + get_datetime("%d,%m,%Y--%H,%M")
    else:
        save_name = model_name + '_' + get_datetime("%d,%m,%Y--%H,%M")

    save_path = Path().resolve().joinpath(
        params['save_dir'],
        params['training_args'].output_dir,
        save_name)

    if save_option:
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("Model saved at: ", save_path)
    else:
        print("Save option is currently disabled. If you wish to keep the current model for future usage, please turn on the saving option by setting save_option=True")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
