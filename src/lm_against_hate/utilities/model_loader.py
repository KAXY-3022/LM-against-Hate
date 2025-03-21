import torch
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TFAutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
from accelerate import init_empty_weights, dispatch_model
from accelerate.utils import infer_auto_device_map

from lm_against_hate.config.config import Causal_params, S2S_params, device, Classifier_params, categories, model_path
from lm_against_hate.utilities.misc import get_datetime


@dataclass
class ModelConfig:
    model_type: str
    model_class: PreTrainedModel
    padding_side: str = 'right'
    peft_model_class: Optional[PreTrainedModel] = None

MODEL_CONFIGS = {
    'Causal': ModelConfig(
        model_type='Causal',
        model_class=AutoModelForCausalLM,
        padding_side='left',
        peft_model_class=AutoPeftModelForCausalLM
    ),
    'S2S': ModelConfig(
        model_type='S2S', 
        model_class=AutoModelForSeq2SeqLM,
        peft_model_class=AutoPeftModelForSeq2SeqLM
    )
}

def model_selection(model_type: str, model_name: Optional[str]) -> dict:
    """Select model parameters based on model type."""
    params_map = {
        'S2S': S2S_params,
        'Causal': Causal_params, 
        'Classifier': Classifier_params
    }
    
    if model_type not in params_map:
        raise ValueError(f'No parameters for model type {model_type} found')
        
    params = params_map[model_type]
    
    if model_name:
        params['model_name'] = model_name
        params['save_name'] = f"{model_name.split('/')[-1]}_againstHate"

    return params


def load_model_with_config(
    model_config: ModelConfig,
    save_dir: Union[str, Path],
    local_model_name: Union[str, Path],
    use_8bit: bool = True,
    use_flash_attention: bool = True,
    device_map: Optional[dict] = 'auto'
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with specified configuration."""
    # Check whether the model is a adapter model
    is_adapter = False
    
    model_path = save_dir.joinpath(local_model_name)
    tokenizer_path = model_path
    adapter_config_path = model_path.joinpath('adapter_config.json')
    
    if adapter_config_path.is_file():
        is_adapter = True
        adapter_config = PeftConfig.from_pretrained(model_path)
        
        adapter_path = model_path
        base_model = adapter_config.base_model_name_or_path.split('/')[-1]
        model_path = save_dir.joinpath(base_model)
        print('base model: ', model_path)
         

    # load tokenizer and model from local path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        padding_side=model_config.padding_side
        )
    
    
    # Init Model 
    try:
        print('loading base model...')
        model = model_config.model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if use_8bit else None,
            attn_implementation="flash_attention_2" if use_8bit and use_flash_attention else None,
            device_map=device_map
        )
    except ValueError as ve:
        # try loading model without flash attention 2
        print(f"Error loading with flash attention: {ve}")
        print("Retrying without flash attention...")
        model = model_config.model_class.from_pretrained(
            model_path,
            device_map=device_map
        )

    except RuntimeError as re:
        # try loading peft model with flash attention 2
        print(f"Error loading model: {re}")
        print("Retrying with PEFT Model...")
        try:
            if model_config.peft_model_class:
                model = model_config.peft_model_class.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16 if use_8bit else None,
                    attn_implementation="flash_attention_2" if use_8bit and use_flash_attention else None,
                    device_map=device_map
                )
            else:
                raise
        except ValueError as ve:
            # try loading model without flash attention 2
            print(f"Error loading with flash attention: {ve}")
            print("Retrying PEFT model without flash attention...")
            if model_config.peft_model_class:
                model = model_config.peft_model_class.from_pretrained(
                    model_path,
                    device_map=device_map
                )
            else:
                raise

    if is_adapter:
        print('Loading adapter model...')
        try:
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, adapter_path)
        except NotImplementedError as nie:
            # Load with meta device
            print(f"Error loading model: {nie}")
            print("Retrying with empty weights...")
            
            with init_empty_weights():
                model = model_config.model_class.from_pretrained(model_path)
                
            device_map = infer_auto_device_map(
                model, max_memory={"cpu": "32GB", "cuda:0": "16GB"})
            dispatch_model(model, device_map, offload_buffers=True)
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, adapter_path)
            
        # Confirm adapter integration
        if hasattr(model.base_model, "active_adapter"):
            print("Active adapter:", model.base_model.active_adapter)
        else:
            print("No adapter applied")
            
    return model, tokenizer

def load_model(
    model_type: str,
    params: dict,
    save: bool = True,
    local_only: bool = False,
    use_8bit: bool = True,
    use_peft: bool = True,
    use_flash_attention: bool = True,
    device_map: Optional[dict] = 'auto'
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model based on type and parameters."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported model type: {model_type}')
        
    model_config = MODEL_CONFIGS[model_type]
    hub_model_name = params['model_name']
    local_model_name = hub_model_name.split('/')[-1]
    if 'save_dir' in params:    
        save_dir = params['save_dir']
    else:
        save_dir = model_path.joinpath(params['model_type'])
        
    peft_config = params.get('peft_config') if use_peft else None

    try:
        model, tokenizer = load_model_with_config(model_config, save_dir, local_model_name, use_8bit, use_flash_attention, device_map)
    except OSError:
        if local_only:
            raise OSError('No local model found')
            
        print('No local model found, downloading model from hub')
        model, tokenizer = download_model_from_hub(
            model_type=model_type,
            model_name=hub_model_name,
            device_map=device_map   
        )
        
        if save:
            save_model(model, tokenizer, save_dir.joinpath(local_model_name))

    if tokenizer.pad_token is None:
        model, tokenizer = add_pad_token(model, tokenizer)
        
    # Add category tokens if needed
    if params['category']:
        print('Adding categories to tokenizer: ', categories)
        model, tokenizer = add_category_tokens(categories, model, tokenizer)
        
        
    if peft_config:
        for param in model.parameters():
            param.requires_grad = False       
            
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    return model, tokenizer

def load_classifiers(
    model_name: str, 
    train: bool = False,
    tf: bool = False,
    save: bool = True,
    num_labels: Optional[int] = None,
    id2label: Optional[dict] = None,
    label2id: Optional[dict] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load classifier models with support for both PyTorch and TensorFlow.
    
    Args:
        model_name: Name of the model to load
        train: Whether to load model in training mode
        tf: Whether to load TensorFlow model
        save: Whether to save model locally after downloading
        num_labels: Number of classification labels. If None, defaults to len(categories)
        id2label: Dictionary mapping label ids to label names. If None, uses default mapping
        label2id: Dictionary mapping label names to label ids. If None, uses default mapping
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_dir = Path().resolve().joinpath('models', 'Classifiers', model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if train:
            if num_labels is None:
                num_labels = len(categories)
            
            if id2label is None or label2id is None:
                # Use default category mappings if not provided
                label2id = {label: str(i) for i, label in enumerate(categories)}
                id2label = {str(i): label for i, label in enumerate(categories)}

            if tf:
                model = TFAutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )     
                           
            # Check if the model is on a meta device
            if not tf:
                if model.device.type == 'meta':
                    # Move the model to an empty state before transferring to a real device
                    model.to_empty()
                    model.to(device)
                else:
                    model.to(device)
        else:
            # Inference mode - load from saved model
            model = TFAutoModelForSequenceClassification.from_pretrained(model_dir) if tf else AutoModelForSequenceClassification.from_pretrained(model_dir, device_map=device)
            
    except OSError:
        # Download model from hub if not found locally
        print('No local model found, downloading model from hub')
        model, tokenizer = download_model_from_hub(
            model_type='Classifier',
            model_name=model_name,
            tf=tf
        )
        model, tokenizer = add_pad_token(model, tokenizer)
        
        if save:
            save_model(model=model, tokenizer=tokenizer, save_path=model_dir)
            
        if train:
            # Recursively load with proper training configuration
            model, tokenizer = load_classifiers(
                model_name=model_name,
                train=True,
                tf=tf,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )

    return model, tokenizer

def load_sentence_transformer(
    model_name: str,
    save: bool = True
) -> SentenceTransformer:
    """Load sentence transformer models."""
    model_dir = model_path.joinpath('Classifiers', 'sentence-transformers', model_name)
    
    try:
        return SentenceTransformer(str(model_dir)).to(device)
    except OSError:
        model = SentenceTransformer(model_name).to(device)
        if save:
            model.save(str(model_dir))
        return model

def download_model_from_hub(
    model_type: str,
    model_name: str,
    tf: bool = False,
    device_map: Optional[dict] = None
) -> Tuple[PreTrainedModel, Optional[PreTrainedTokenizer]]:
    """Download model from Hugging Face Hub."""
    if model_type == "Classifier":
        model_class = (TFAutoModelForSequenceClassification if tf 
                      else AutoModelForSequenceClassification)
        model = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
        
    if model_type == "Transformer":
        return SentenceTransformer(model_name).to(device), None
        
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported model type: {model_type}')
        
    config = MODEL_CONFIGS[model_type]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side=config.padding_side
    )
    
    try:
        model = config.model_class.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    except ValueError:
        model = config.model_class.from_pretrained(
            model_name,
            device_map=device
        )
        
    return model, tokenizer

def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: Path
) -> None:
    """Save model and tokenizer locally."""
    print('Saving model locally for future usage at: ', save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def save_trained_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: dict,
    save_option: bool = True,
    targetawareness: bool = False,
    is_classifier: bool = False
) -> None:
    """Save trained model and tokenizer."""
    if not save_option:
        print("Save option disabled. Enable with save_option=True to keep model.")
        return

    if is_classifier:
        model_name = params['model_name']
        save_name = (f"{model_name.split('/')[0]}/target_demographic_"
                    f"{model_name.split('/')[-1]}")
    else:
        model_name = params['save_name'].split('/')[-1]
        category_suffix = '-category' if targetawareness else ''
        timestamp = get_datetime("%d,%m,%Y--%H,%M")
        save_name = f"{model_name}{category_suffix}_{timestamp}"

    save_path = Path().resolve().joinpath(params['save_dir'], save_name)
    save_model(model, tokenizer, save_path)

def add_pad_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Add padding token if not present."""
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def add_category_tokens(
    types: list,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Add target demographic tokens to model vocabulary."""
    new_tokens = {f'<{target}>' for target in types}
    new_tokens -= set(tokenizer.get_vocab().keys())
    
    if new_tokens:
        print(new_tokens)
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

from os import walk
# TODO finish loading all models from directory
def get_inference_model_list(model_path):
    model_types = ['S2S', 'Causal']
    model_list = []
    for _type in model_types:
        new_path = model_path.joinpath(_type)
        for (dirpath, dirnames, filenames) in walk(new_path):
            for dir in dirnames:
                model_config = {
                    "model_type": _type,
                    "model_name": dir,
                    "category": 'category' in dir
                }
                model_list.append(model_config)
            break
    return model_list