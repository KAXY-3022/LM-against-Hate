import torch
from pathlib import Path
from typing import Optional
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TFAutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM)
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM

from ..config.config import Causal_params, S2S_params, device, Classifier_params, categories
from ..utilities.misc import get_datetime


def model_selection(model_type: str, model_name: Optional[str]):
    if model_type == 'S2S':
        params = S2S_params
    elif model_type == 'Causal':
        params = Causal_params
    elif model_type == 'Classifier':
        params = Classifier_params
    else:
        raise ValueError('no parameters for given modeltype {modeltype} found')
    if model_name:
        params['model_name'] = model_name
        params['save_name'] = model_name.split('/')[-1] + '_againstHate'

    return params


def load_model(model_type: str,
               params: dict,
               save: bool = True,
               local_only: bool = True):
    """
    model loader for different model types.

    Args:
        model_type: Type of model ('Causal', 'S2S').
        params: Dictionary of model parameters ('model_name', 'save_dir', 'peft_config')
        save: Whether to save the model locally if loaded from the hub.
        local_only: Whether to only load model locally

    Returns:
        Tuple of (model, tokenizer).
    """
    # Load base model from
    hub_model_name = params['model_name']
    local_model_name = params['model_name'].split('/')[-1]
    local_dir = params['save_dir'].joinpath(local_model_name)
    peft_config = params.get('peft_config', None)
    _model_type = None

    if model_type == "Causal":
        _model_type = AutoModelForCausalLM
        padding_side = 'left'
    elif model_type == "S2S":
        _model_type = AutoModelForSeq2SeqLM
        padding_side = 'right'

    print('Loading local model from: ', local_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir, padding_side=padding_side)
        model = _model_type.from_pretrained(
            local_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device)

    except RuntimeError:
        if model_type == "Causal":
            _model_type = AutoPeftModelForCausalLM
        elif model_type == "S2S":
            _model_type = AutoPeftModelForSeq2SeqLM
        # if local model is peft model, use AutoPeftModel
        model = _model_type.from_pretrained(
            local_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device)

    except ValueError:
        # If no support for flash_attention_2, then use standard implementation
        model = _model_type.from_pretrained(local_dir, device_map=device)

    except OSError:
        if local_only:
            raise OSError('No local model found')
        else:
            print('No local model found, downloading model from hub')
            model, tokenizer = download_model_from_hub(
                model_type=model_type,
                model_name=hub_model_name,
                device=device)
            if save:
                save_model(model, tokenizer, local_dir)

    if tokenizer.pad_token is None:
        model, tokenizer = add_pad_token(model, tokenizer)

    if peft_config:
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    return model, tokenizer


def load_classifiers(model_name: str, train: bool = True, tf: bool = False, save: bool = True):
    """
    Model loader for classifier models.

    Args:
        model_name: Name of the model to load.
        tf: Whether to load TensorFlow-based models.
        save: Whether to save the model locally if loaded from the hub.

    Returns:
        A tuple of (model, tokenizer).
    """

    # initialize model and tokenizer
    model_dir = Path().resolve().joinpath('models', 'Classifiers', model_name)

    try:
        # load local model if exists
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if train:
            # create label2id, id2label dicts for nice outputs for the model
            num_labels = len(categories)
            label2id, id2label = dict(), dict()
            for i, label in enumerate(categories):
                label2id[label] = str(i)
                id2label[str(i)] = label
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                device_map=device
            )
        else:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_dir) if tf else AutoModelForSequenceClassification.from_pretrained(model_dir, device_map=device)
    except OSError:
        # load model from source
        model, tokenizer = download_model_from_hub(
            model_type='Classifier', model_name=model_name, device=device, tf=tf)
        model, tokenizer = add_pad_token(model, tokenizer)
        if save:
            # save model locally for future usage
            save_model(model=model, tokenizer=tokenizer, save_path=model_dir)
        if train:
            # load the downloaded model with correct label2id, id2label dicts
            model, tokenizer = load_classifiers(
                model_name=model_name, train=True)

    return model, tokenizer


def load_sentence_transformer(model_name: str, save: bool = True) -> SentenceTransformer:
    """
    Model loader for classifier models.

    Args:
        model_name: Name of the model to load.
        save: Whether to save the model locally if loaded from the hub.

    Returns:
        A tuple of (model, tokenizer).
    """

    # initialize model
    model_path = str(Path().resolve().joinpath(
        'models', 'Classifiers', 'sentence-transformers', model_name))
    try:
        # load local model if exists
        return SentenceTransformer(model_path).to(device)
    except OSError:
        # load model from source
        model = SentenceTransformer(model_name).to(device)
        if save:
            # save model locally for future use
            model.save(str(model_path))
        return model


def download_model_from_hub(model_type: str, model_name: str, tf: bool = False):
    """
    Download model from the Hugging Face Hub based on model type.

    Args:
        model_type: Type of model to download.
        model_name: Name of the model to download.
        tf: Whether the model is tensorflow model

    Returns:
        Loaded model and tokenizer.
    """
    _model_type = None
    if model_type == "Causal":
        _model_type = AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left')

    elif model_type == "S2S":
        _model_type = AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_type == "Classifier":
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name) if tf else AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, AutoTokenizer.from_pretrained(model_name)

    elif model_type == "Transformer":
        return SentenceTransformer(model_name).to(device), None

    try:
        model = _model_type.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device)
    except ValueError:
        # If no support for flash_attention_2, then use standard implementation
        model = _model_type.from_pretrained(
            model_name, device_map=device)
    return model, tokenizer


def save_model(model, tokenizer, save_path: Path):
    """
    Save model and tokenizer locally.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        save_path: Directory to save the model and tokenizer.
    """
    print('Saving model locally for future usage at: ', save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def save_trained_model(model, tokenizer, params, save_option=True, targetawareness: bool = False, is_classifier: bool = False):
    '''
    Save model and tokenizer to a local directory.
    
    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        params: Parameters for the model.
        save_option: Whether to save the model.
        targetawareness: Whether the model is trained with target demographic embeddings.      
        is_classifier: Whether the model is a classifier.
    '''
    if not save_option:
        print("Save option is currently disabled. If you wish to keep the current model for future usage, please turn on the saving option by setting save_option=True")
        return

    # save model and tokenizer to a local directory.
    if is_classifier:
        model_name = params['model_name']
        save_name = model_name.split(
            '/')[0] + 'target_demographic_' + model_name.split('/')[-1]

    else:
        model_name = params['save_name'].split('/')[-1]
        if targetawareness:
            save_name = model_name + '-category' + \
                '_' + get_datetime("%d,%m,%Y--%H,%M")
        else:
            save_name = model_name + '_' + get_datetime("%d,%m,%Y--%H,%M")

    save_path = Path().resolve().joinpath(
        params['save_dir'],
        save_name)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("Model saved at: ", save_path)


def add_pad_token(model, tokenizer):
    '''
    Add pad token to the tokenizer and model
    
    Args:
        model: Model to add pad token to.
        tokenizer: Tokenizer to add pad token to.

    Returns:
        Model and tokenizer with pad token added.
    '''
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def add_category_tokens(types: list, model, tokenizer):
    '''
    Add each target demographic as a new token to tokenizer
    <CLASS1> <CLASS2> etc.
    
    Args:
        types: List of target demographics.
        model: Model to add target demographic tokens to.
        tokenizer: Tokenizer to add target demographic tokens to.

    Returns:
        Model and tokenizer with target demographic tokens added.
    '''
    new_tokens = []
    for target in types:
        new_tokens.append('<' + target + '>')
    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
