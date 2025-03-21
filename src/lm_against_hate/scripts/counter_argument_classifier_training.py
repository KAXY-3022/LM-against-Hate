from lm_against_hate.utilities.model_loader import model_selection, save_trained_model, load_classifiers
from lm_against_hate.utilities.DataLoader import ClassifierDataLoader
from lm_against_hate.utilities.cleanup import cleanup_resources
from transformers import EarlyStoppingCallback, Trainer


def main(model_name, resume_from_checkpoint):
    """
    Train a counter-argument classifier model using TensorFlow.
    
    Args:
        model_name: Name/path of the pretrained model to fine-tune
        resume_from_checkpoint: Whether to resume training from a checkpoint
    """

    # Load Model Parameters
    print(f'Loading model parameters for: {model_name}')
    params = model_selection(model_type='Classifier', model_name=model_name)


    # Define label mappings for counter-argument classification
    num_labels = 2
    id2label = {
        0: "non-counter-argument",
        1: "counter-argument"
    }
    label2id = {
        "non-counter-argument": 0,
        "counter-argument": 1
    }
    
    # Store label config in params for DataLoader and model saving
    params['id2label'] = id2label
    params['label2id'] = label2id
    params['num_labels'] = num_labels
    params['tf'] = False

    # Load model and tokenizer
    print("Loading tokenizer and model...")
    model, tokenizer = load_classifiers(
        model_name=model_name,
        train=True,
        tf=params['tf'],  # Using TensorFlow for counter-argument classification
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
    
    trainer = Trainer(
        model=model,
        args=params['training_args'],
        train_dataset=dataloader.ds['train'],
        eval_dataset=dataloader.ds['val'],
        tokenizer=tokenizer,
        data_collator=dataloader.datacollator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            )
        ]
    )
    
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
        model_name="chkla/roberta-argument",
        resume_from_checkpoint=False
    )
    
    # chkla/roberta-argument
