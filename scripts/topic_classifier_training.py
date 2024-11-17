# set data paths
import os
import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from pathlib import Path
from transformers import EvalPrediction, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from param import categories
from utilities import get_datetime

def main(model_name):
    # set data paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = Path().resolve().joinpath('models', 'Classifiers')
    train_dir = Path().resolve().joinpath('data', 'Custom', 'Topic-classification_train.csv')
    test_dir = Path().resolve().joinpath('data', 'Custom', 'Topic-classification_test.csv')

    df_raw = pd.read_csv(train_dir)
    df_new = df_raw.copy()

    df_new = df_new.dropna()
    df_new = df_new[categories]
    for cat in categories:
        df_new[cat] = df_new[cat].astype(float)

    ds = Dataset.from_pandas(df_new)
    dataset = ds.train_test_split(test_size=0.2)
    dataset = dataset.remove_columns("__index_level_0__")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # create labels column
    cols = dataset["train"].column_names
    dataset = dataset.map(lambda x : {"labels": [x[c] for c in cols if c != "text"]})

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt')


    cols=dataset["train"].column_names
    cols.remove("labels")
    tokenized_ds = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=cols
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    )



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