import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from lm_against_hate.utilities.model_loader import model_selection, load_classifiers
from lm_against_hate.utilities.DataLoader import ClassifierDataLoader
from lm_against_hate.utilities.cleanup import cleanup_resources
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def main(model_name):
    """
    Test a counter-argument classifier model's performance.
    
    Args:
        model_name: Name/path of the pretrained model to test
    """
    # Load Model Parameters
    print(f'Loading model parameters for: {model_name}')
    params = model_selection(model_type='Classifier', model_name=model_name)

    # Define label mappings
    num_labels = 2
    id2label = {
        0: "non-counter-argument",
        1: "counter-argument"
    }
    label2id = {
        "non-counter-argument": 0,
        "counter-argument": 1
    }
    
    params['id2label'] = id2label
    params['label2id'] = label2id
    params['num_labels'] = num_labels
    params['tf'] = True

    # Load model and tokenizer
    print("Loading tokenizer and model...")
    model, tokenizer = load_classifiers(
        model_name=model_name,
        train=False,
        tf=params['tf'],
        save=False,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Load test data
    print("Loading and preparing test data...")
    dataloader = ClassifierDataLoader(param=params, tokenizer=tokenizer)
    dataloader.load_test_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)

    # Prepare TF dataset
    batch_size = int(params['training_args'].per_device_eval_batch_size)
    tf_dataset_test = model.prepare_tf_dataset(
        dataloader.ds["test"],
        batch_size=batch_size,
        shuffle=False,
        tokenizer=tokenizer
    )

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(tf_dataset_test)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    
    # Get true labels
    true_labels = [example['labels'] for example in dataloader.ds['test']]
    
    # Calculate metrics
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, 
                              target_names=list(label2id.keys())))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    
    # Calculate ROC-AUC score
    probabilities = tf.nn.softmax(predictions.logits, axis=1)
    roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Cleanup
    cleanup_resources()

if __name__ == "__main__":
    main(model_name="tum-nlp/bert-counterspeech-classifier")
