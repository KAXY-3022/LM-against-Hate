from transformers import TrainingArguments, Seq2SeqTrainingArguments
from pathlib import Path
from peft import LoraConfig, TaskType

# selected demographics
categories = ["MIGRANTS", "POC", "LGBT+", "MUSLIMS", "WOMEN", "JEWS", "other", "DISABLED"]

def optuna_hp_space(trial):
  return {
      "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
      "warmup_ratio": trial.suggest_float("warmup_ratio", 0.1, 0.3, log=True),
      #"num_train_epochs":trial.suggest_int('num_train_epochs', low = 3, high = 8),
      #"per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
      "weight_decay":trial.suggest_float('weight_decay', 0.01, 0.3),
      }

Causal_training_args = TrainingArguments(
    output_dir="Causal",
    num_train_epochs=10.0,
    learning_rate=3e-05,
    weight_decay=0.05,
    warmup_ratio=0.1,
    optim="adamw_bnb_8bit",
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3, 
    load_best_model_at_end=True,
    report_to="none",
    torch_compile=False,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    auto_find_batch_size=True,
    # per_device_train_batch_size=1,
    logging_dir=Path().resolve().joinpath("logs"),
)

Causal_params = {
  'model_name': "openai-community/gpt2-medium",
  'save_name': "gpt2-medium_againstHate",
  'train_dir': Path().resolve().joinpath('data', 'Custom', 'CONAN_train.csv'),
  'save_dir': Path().resolve().joinpath('models'),
  'training_args': Causal_training_args,
  'category': False,
  'peft_config': LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
}

S2S_training_args = Seq2SeqTrainingArguments(
    output_dir="S2S",
    num_train_epochs=10.0,
    learning_rate=3e-5,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    warmup_ratio=0.2,
    optim="adamw_bnb_8bit",
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    tf32=True,
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    # per_device_train_batch_size=1,
    logging_dir=Path().resolve().joinpath("logs"),
)

S2S_params = {
  'model_name': 'facebook/bart-large',
  'save_name': "bart-large_againstHate",
  'train_dir': Path().resolve().joinpath('data', 'Custom', 'CONAN_train.csv'),
  'save_dir': Path().resolve().joinpath('models'),
  'training_args': S2S_training_args,
  'category': False,
  'peft_config': LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
}