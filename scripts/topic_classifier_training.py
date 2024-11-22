# set data paths
import os
import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from pathlib import Path
from transformers import EvalPrediction, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from param import categories, data_path_new, model_path
from utilities.utils import get_datetime, add_pad_token

def main(model_name):
    # set data paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = model_path.joinpath('Classifiers')
    train_dir = data_path_new.joinpath('Classifier_train.csv')
    val_dir = data_path_new.joinpath('Classifier_val.csv')

    df_train = pd.read_csv(train_dir)
    df_val = pd.read_csv(val_dir)

    df_train_labels = df_train.dropna()
    df_train_labels = df_train_labels[categories]
    print(df_train_labels)
    for cat in categories:
        df_train_labels[cat] = df_train_labels[cat].astype(float)

    dataset = Dataset.from_pandas(df_train_labels)
    dataset = dataset.remove_columns("__index_level_0__")

    # create labels column
    cols = dataset.column_names
    dataset = dataset.map(
        lambda x: {"labels": [x[c] for c in cols if c != "text"]})

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt')
        
    cols = dataset.column_names
    cols.remove("labels")
    tokenized_ds = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=cols
    )
    
    num_labels = len(categories)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(categories):
        label2id[label] = str(i)
        id2label[str(i)] = label



    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    tokenizer, model = add_pad_token(tokenizer, model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create label2id, id2label dicts for nice outputs for the model
    





    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.6):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    training_args = TrainingArguments(
        num_train_epochs= 8,
        output_dir= os.path.join(model_dir, model_name),
        learning_rate=1e-5,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=False,)
    pt_save_directory = os.path.join(model_dir, model_name.replace('/', '-'))
    pt_save_directory = os.path.join(pt_save_directory, get_datetime("%d,%m,%Y--%H,%M"))

    tokenizer.save_pretrained(pt_save_directory)
    model.save_pretrained(pt_save_directory)

    
if __name__ == "__main__":
    main(model_name="cardiffnlp/tweet-topic-21-multi")
    # NLP-LTU/bertweet-large-sexism-detector
    # cardiffnlp/tweet-topic-21-multi
    # cardiffnlp/tweet-topic-latest-multi