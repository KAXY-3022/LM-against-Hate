import torch
import numpy as np
from typing import Dict, Any
import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# Suppress specific TensorFlow deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    'ignore', category=DeprecationWarning, module='tensorflow')

from transformers import (
    EvalPrediction,
    EarlyStoppingCallback,
    Trainer
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from lm_against_hate.utilities.model_loader import load_classifiers, model_selection, save_trained_model
from lm_against_hate.utilities.DataLoader import ClassifierDataLoader
from lm_against_hate.utilities.cleanup import cleanup_resources
from lm_against_hate.config.config import categories

class CustomTrainer(Trainer):
    """Custom trainer with weighted loss and gradient clipping."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights).to(logits.device)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                       labels.float().view(-1, self.model.config.num_labels))
        
        return (loss, outputs) if return_outputs else loss

def main(model_name, resume_from_checkpoint):
    """
    Train a topic classifier model.
    
    Args:
        model_name: Name/path of the pretrained model to fine-tune
        resume_from_checkpoint: Whether to resume training from a checkpoint
    """
    # Load Model Parameters
    print(f'Loading model parameters for: {model_name}')
    params = model_selection(model_type='Classifier', model_name=model_name)

    # Define label mappings for topic classification
    num_labels = len(categories)
    label2id = {label: str(i) for i, label in enumerate(categories)}
    id2label = {str(i): label for i, label in enumerate(categories)}

    # Store label config in params for DataLoader and model saving
    params['id2label'] = id2label
    params['label2id'] = label2id
    params['num_labels'] = num_labels

    # Load model and tokenizer
    print("Loading tokenizer and model...")
    model, tokenizer = load_classifiers(
        model_name=model_name,
        train=True,
        tf=False,  # Using PyTorch for topic classification
        save=True,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Load data
    print("Loading and preparing data...")
    dataloader = ClassifierDataLoader(param=params, tokenizer=tokenizer)
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            p: EvalPrediction object containing predictions and labels
            
        Returns:
            Dictionary of computed metrics
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = torch.sigmoid(torch.tensor(preds))
        y_pred = (probs >= 0.5).numpy().astype(int)
        y_true = p.label_ids
        
        # Compute metrics
        f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Compute per-class metrics
        per_class_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        class_metrics = {
            f'f1_{categories[i]}': score 
            for i, score in enumerate(per_class_f1)
        }
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            **class_metrics
        }

    # Calculate class weights for balanced training
    total_samples = len(dataloader.ds['train'])
    class_samples = np.sum(dataloader.ds['train']['labels'], axis=0)
    class_weights = torch.tensor([
        total_samples / (num_labels * count) if count > 0 else 1.0
        for count in class_samples
    ])

    # Initialize trainer with optimizations
    trainer = CustomTrainer(
        model=model,
        args=params['training_args'],
        train_dataset=dataloader.ds['train'],
        eval_dataset=dataloader.ds['val'],
        tokenizer=tokenizer,
        data_collator=dataloader.datacollator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,     
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Train model
    print("Training starts...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save trained model
    save_trained_model(
        model, 
        tokenizer, 
        params, 
        save_option=True, 
        targetawareness=False, 
        is_classifier=True
    )
    
    # Cleanup
    cleanup_resources()

if __name__ == "__main__":
    main(
        model_name="NLP-LTU/bertweet-large-sexism-detector",
        resume_from_checkpoint=False
    )
    
    # NLP-LTU/bertweet-large-sexism-detector
    # cardiffnlp/tweet-topic-21-multi
    # cardiffnlp/tweet-topic-latest-multi