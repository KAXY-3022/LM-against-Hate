from transformers import TrainingArguments, Seq2SeqTrainingArguments
from pathlib import Path
from peft import LoraConfig, TaskType
import torch

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
        learning_rate=3e-5,
        weight_decay=0.05,
        warmup_ratio=0.1,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        auto_find_batch_size=True,
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
    'peft_config': LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
    'training_args': TrainingArguments(
        output_dir=model_path.joinpath('Causal'),
        num_train_epochs=200.0,
        learning_rate=3e-05,
        weight_decay=0.05,
        warmup_ratio=0.1,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        torch_compile=False,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        auto_find_batch_size=True,
        # per_device_train_batch_size=1,
        logging_dir=Path().resolve().joinpath("logs"),
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
    'peft_config': LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
    'training_args': Seq2SeqTrainingArguments(
        output_dir=model_path.joinpath('S2S'),
        num_train_epochs=200.0,
        learning_rate=3e-5,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        warmup_ratio=0.2,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        tf32=True,
        load_best_model_at_end=True,
        auto_find_batch_size=True,
        # per_device_train_batch_size=1,
        logging_dir=Path().resolve().joinpath("logs"),
    ),
}