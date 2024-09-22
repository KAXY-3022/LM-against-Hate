from transformers import TrainingArguments, Seq2SeqTrainingArguments
from pathlib import Path

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

GPT2_training_args = TrainingArguments(
    output_dir="GPT2",
    num_train_epochs=10.0,
    learning_rate=3.800568576836524e-05,
    weight_decay=0.050977894796868116,
    warmup_ratio=0.10816909354342182,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3, 
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    report_to="none",
    logging_dir=Path().resolve().joinpath("logs"),
)

GPT2_params = {
  'model_name': "gpt2-medium",
  'save_name': "gpt2_CONAN",
  'train_dir': Path().resolve().joinpath('data', 'Custom', 'CONAN_train.csv'),
  'save_dir': Path().resolve().joinpath('models'),
  'training_args': GPT2_training_args,
  'category': True
}

BART_training_args = Seq2SeqTrainingArguments(
    output_dir="BART",
    num_train_epochs=10.0,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.2,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    logging_dir=Path().resolve().joinpath("logs"),
)

BART_params = {
  'model_name': 'facebook/bart-large',
  'save_name': "bart_CONAN",
  'train_dir': Path().resolve().joinpath('data', 'Custom', 'CONAN_train.csv'),
  'save_dir': Path().resolve().joinpath('models'),
  'training_args': BART_training_args,
  'category': True
}