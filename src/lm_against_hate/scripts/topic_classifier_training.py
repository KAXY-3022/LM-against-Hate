# set data paths
import os
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from transformers import EvalPrediction, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from ..utilities.misc import get_datetime
from ..utilities.model_loader import add_pad_token, load_classifiers, model_selection, save_trained_model
from ..utilities.DataLoader import ClassifierDataLoader

def main(model_name, resume_from_checkpoint):
    # Load Model Parameters
    print(f'Loading model parameters for: {model_name}')
    params = model_selection(model_type='Classifier', model_name=model_name)

    # load model
    model, tokenizer = load_classifiers(model_name=model_name,train=True)
    model, tokenizer = add_pad_token(model=model, tokenizer=tokenizer)
    
    # load data
    dataloader = ClassifierDataLoader(params=params, tokenizer=tokenizer)
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)


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



    # train
    print('training starts')
    trainer = Trainer(
        model=model,
        args=params['training_args'],
        train_dataset=dataloader.ds['train'],
        eval_dataset=dataloader.ds['val'],
        processing_class=tokenizer,
        data_collator=dataloader.datacollator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    save_trained_model(model, tokenizer, params, save_option=True, targetawareness=False, is_classifier=True)

    
if __name__ == "__main__":
    main(model_name="cardiffnlp/tweet-topic-21-multi",
         resume_from_checkpoint=False)
    # NLP-LTU/bertweet-large-sexism-detector
    # cardiffnlp/tweet-topic-21-multi
    # cardiffnlp/tweet-topic-latest-multi