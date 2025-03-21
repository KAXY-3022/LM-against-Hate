import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

from lm_against_hate.utilities.DataLoader import ClassifierDataLoader
from lm_against_hate.utilities.model_loader import load_classifiers
from lm_against_hate.utilities.batch_processing import batchify
from lm_against_hate.config.config import categories, Classifier_params, device, model_path

def test_model(model_name: str) -> None:
    """
    Test a single topic classifier model.
    
    Args:
        model_name: Name/path of the classifier model to test
    """
    print(f"\nTesting model: {model_name}")
    model, tokenizer = load_classifiers(
        model_name=model_name,
        train=False,
    )
    
    # Prepare parameters and data
    params = Classifier_params.copy()
    params['model_name'] = model_name
    params['num_labels'] = len(categories)
    params['tf'] = False
    dataloader = ClassifierDataLoader(param=params, tokenizer=tokenizer)
    
    # Load test data from the default test path in config
    try:
        print("loading test data...")
        dataloader.load_test_data()
        dataloader.eval()
        dataloader.prepare_dataset(tokenizer=tokenizer)
        test_data = dataloader.ds['test']
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return
    
    # Get predictions
    all_predictions = []
    model.eval()
    batch_size = 32
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 512
    
    with torch.no_grad():
        for batch_texts in batchify(test_data['text'], batch_size, "Processing batches"):
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            batch_preds = (probs >= 0.5).int()
            all_predictions.append(batch_preds)
    predictions = torch.cat(all_predictions, dim=0).cpu()
    
    # Convert label columns to binary format
    true_labels = torch.tensor([[float(val) for val in row] for row in test_data['labels']])
    
    # Generate classification report
    report = classification_report(
        y_true=true_labels,
        y_pred=predictions,
        target_names=categories,
        zero_division=0
    )
    
    print("\nClassification Report:")
    print(report)

def main():
    """Test all topic classifier models under Classifiers/cardiffnlp and Classifiers/NLP-LTU directories."""
    model_paths = []
    
    # Get models from cardiffnlp directory
    cardiff_dir = model_path.joinpath('Classifiers', 'cardiffnlp')
    if cardiff_dir.exists():
        model_paths.extend([str(d) for d in cardiff_dir.iterdir() if d.is_dir()])
        
    # Get models from NLP-LTU directory
    nlp_ltu_dir = model_path.joinpath('Classifiers', 'NLP-LTU')
    if nlp_ltu_dir.exists():
        model_paths.extend([str(d) for d in nlp_ltu_dir.iterdir() if d.is_dir()])
    
    if not model_paths:
        print("No models found in Classifiers/cardiffnlp or Classifiers/NLP-LTU directories")
        return
        
    print(f"Found {len(model_paths)} models to test")
    
    # Test each model
    for model_path_str in model_paths:
        try:
            test_model(model_path_str)
        except Exception as e:
            print(f"Error testing model {model_path_str}: {str(e)}")

if __name__ == "__main__":
    main()
