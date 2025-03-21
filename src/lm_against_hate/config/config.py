from transformers import TrainingArguments, Seq2SeqTrainingArguments
from pathlib import Path
from peft import LoraConfig, TaskType
import torch
from accelerate import accelerator

# Check Device Availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# selected demographics
categories = ["MIGRANTS", "POC", "LGBT+",
              "MUSLIMS", "WOMEN", "JEWS", "other", "DISABLED"]

# some path to use
data_path = Path().resolve().joinpath('data', 'Custom')
data_path_new = Path().resolve().joinpath('data', 'Custom_New')
model_path = Path().resolve().joinpath('models')
pred_dir = Path().resolve().joinpath('predictions')

# system prompt
system_message = {
    'role': 'system',
    'content': 'You are a UN agent who focuses on fighting online hate speech. Your task is to write responses, so-called counter speech, to online hate speech targeting different demographics. The responses should not be offensive, hateful, or toxic. The responses should actively fight against the hateful messages and bring counterarguments to the table. The aim is to bring in positive perspectives, clarify misinformation, and be an active voice for the targeted demographics against hate.'}


def optuna_hp_space(trial):
  return {
      "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
      "warmup_ratio": trial.suggest_float("warmup_ratio", 0.1, 0.3, log=True),
      # "num_train_epochs":trial.suggest_int('num_train_epochs', low = 3, high = 8),
      # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
      "weight_decay": trial.suggest_float('weight_decay', 0.01, 0.3),
  }

Classifier_params = {
    'model_name': "cardiffnlp/tweet-topic-21-multi",
    'save_name': "cardiffnlp/target-demographic-21-multi",
    'model_type': "Classifier",
    'train_dir': data_path_new.joinpath('Classifier_train.csv'),
    'val_dir': data_path_new.joinpath('Classifier_val.csv'),
    'test_dir': data_path_new.joinpath('Classifier_test.csv'),
    'save_dir': model_path.joinpath('Classifiers'),
    'training_args': TrainingArguments(
        output_dir=model_path.joinpath('Classifiers'),
        num_train_epochs=200.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy='epoch',
        save_strategy='epoch', 
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        per_device_train_batch_size=128,
        max_grad_norm=1.0,
        fp16=True,
        group_by_length=True,
        report_to='none', 
    )
}

Causal_params = {
    'model_name': "openai-community/gpt2-medium",
    'save_name': "gpt2-medium_againstHate",
    'model_type': "Causal",
    'train_dir': data_path.joinpath('CONAN_train.csv'),
    'val_dir': data_path.joinpath('CONAN_val.csv'),
    'test_dir': data_path.joinpath('CONAN_test.csv'),
    'save_dir': model_path.joinpath('Causal'),
    'category': False,
    'peft_config': LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        modules_to_save=['embed_token'],
        target_modules=["self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj", "self_attn.o_proj", "embed_token"],
        task_type="CAUSAL_LM"
    ),
    'training_args': TrainingArguments(
        output_dir=model_path.joinpath('Causal'),
        num_train_epochs=100.0,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_smoothing_factor=0.1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        auto_find_batch_size=True,
        logging_dir=Path().resolve().joinpath("logs"),
        logging_steps=1000,
        log_level="info",
        report_to="none",
    ),
}


S2S_params = {
    'model_name': 'facebook/bart-large',
    'save_name': "bart-large_againstHate",
    'model_type': "S2S",
    'train_dir': data_path.joinpath('CONAN_train.csv'),
    'val_dir': data_path.joinpath('CONAN_val.csv'),
    'test_dir': data_path.joinpath('CONAN_test.csv'),
    'save_dir': model_path.joinpath('S2S'),
    'category': False,
    'peft_config': LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="SEQ_2_SEQ_LM"
    ),
    'training_args': Seq2SeqTrainingArguments(
        output_dir=model_path.joinpath('S2S'),
        num_train_epochs=290,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        bf16=True,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        auto_find_batch_size=True,
        logging_dir=Path().resolve().joinpath("logs"),
        logging_steps=50,
        log_level="info",
        report_to="none",
    ),
}
