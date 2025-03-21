import re
import statistics
from typing import List, Dict, Any
import torch
import tensorflow as tf
from sentence_transformers import util
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pandas import DataFrame
from googleapiclient import discovery
import httplib2
import time
import json

from lm_against_hate.utilities.cleanup import cleanup_resources
from lm_against_hate.utilities.batch_processing import batchify
from lm_against_hate.utilities.model_loader import load_classifiers, load_sentence_transformer
from lm_against_hate.config.eval_config import MODEL_PATHS

def compute_similarity_pipeline(dfs: List[DataFrame], model_paths: List[str], task: str) -> np.ndarray:
    """
    Compute similarity scores using sentence transformers.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        model_paths: List of sentence-transformer model paths to use for evaluation.
        task: Task type, either 'label' or 'context'.

    Returns:
        np.ndarray: 2D array of similarity scores [n_datasets, n_models].
    """
    scores = np.zeros([len(dfs), len(model_paths)])
    __ = [None for _ in range(len(model_paths))]
    scores_per_input = [__ for _ in range(len(dfs))]
    
    for model_idx, model_name in enumerate(model_paths):
        print('-'*80)
        print(f"Computing {task} Similarity Score with {model_name}")
        model = load_sentence_transformer(model_name)

        for df_idx, df in enumerate(dfs):
            print(f"{df_idx+1}/{len(dfs)}")
            column = "Counter_Speech" if task == 'label' else "Hate_Speech"
            inputs = df[column].tolist()
            preds = df["Prediction"].tolist()
            
            cosine_average, cosine = compute_similarity_score(model, preds, inputs)
            scores[df_idx][model_idx] = cosine_average
            scores_per_input[df_idx][model_idx] = cosine
            cleanup_resources()

    return scores, scores_per_input


def compute_similarity_score(model, preds: List[str], targets: List[str]) -> float:
    """
    Compute cosine similarity scores between predictions and targets using a sentence transformer model.

    Args:
        model: Sentence transformer model for encoding text into embeddings.
        preds: List of predicted text strings to compare.
        targets: List of target text strings to compare against.

    Returns:
        float: Mean normalized cosine similarity score between predictions and targets.
               Scores are normalized from [-1,1] to [0,1] range where:
               - 1.0 indicates identical semantic meaning
               - 0.5 indicates unrelated meaning 
               - 0.0 indicates opposite meaning
    """
    preds_embeddings = model.encode(preds, show_progress_bar=True)
    targets_embeddings = model.encode(targets, show_progress_bar=True)

    cosine_score = util.cos_sim(preds_embeddings, targets_embeddings).diagonal()
    # Normalize cosine similarity from [-1,1] to [0,1]
    normalized_cosine_score = (cosine_score + 1)/2
    return statistics.fmean(normalized_cosine_score), normalized_cosine_score.tolist()


def compute_toxicity_score(dfs: List[DataFrame], model_path: str, args: Dict[str, Any], soft: bool = False) -> np.ndarray:
    """
    Calculate toxicity scores for predictions using a pre-trained model.
    Model trained on merged English Jigsaw datasets (~2M examples).
    RoBERTa model achieves AUC-ROC of 0.98 and F1-score of 0.76 on Jigsaw test set.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        model_path: Path to the pre-trained toxicity model.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
        soft: If True, return raw probabilities. If False, return binary predictions.

    Returns:
        np.ndarray: List of toxicity scores for each dataset (1=neutral, 0=toxic).
    """
    print('-'*80)
    print(f"Computing toxicity scores with {model_path}")
    model, tokenizer = load_classifiers(model_path)
    scores = []
    scores_per_input = []

    for df_idx, df in enumerate(dfs):
        results = []
        preds = df["Prediction"].tolist()
        
        for batch in batchify(preds, args["batch_size"], description=f"{df_idx+1}/{len(dfs)}"):
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, -1)[:, 1].cpu().numpy() if soft else (logits[:, 1] > 0.6).cpu().numpy()
                results.extend([1 - p for p in probabilities])
        scores_per_input.append(results)
        scores.append(statistics.fmean(results))
        cleanup_resources()
        
    return np.array(scores), scores_per_input


def compute_cola_score(dfs: List[DataFrame], args: Dict[str, Any], soft: bool = False) -> np.ndarray:
    """
    Calculate grammatical acceptability (CoLA) scores for predictions.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
        soft: If True, return raw probabilities. If False, return binary predictions.

    Returns:
        np.ndarray: List of CoLA scores for each dataset (1=acceptable, 0=unacceptable).
    """
    print('-'*80)
    print(f"Computing CoLA scores with {MODEL_PATHS['cola']}")
    model, tokenizer = load_classifiers(MODEL_PATHS['cola'])
    scores = []
    score_per_input = []

    for df_idx, df in enumerate(dfs):
        results = []
        preds = df["Prediction"].tolist()
        
        for batch in batchify(preds, args["batch_size"], description=f"{df_idx+1}/{len(dfs)}"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, -1)[:, 0].cpu().numpy()
                results.extend(probabilities if soft else (probabilities > 0.5).tolist())
                
        score_per_input.append(results)
        scores.append(1 - statistics.fmean(results))
        cleanup_resources()
        
    return np.array(scores), score_per_input


def compute_Offense_Hate_score(dfs: List[DataFrame], args: Dict[str, Any], soft: bool = False) -> List[Dict[str, np.ndarray]]:
    """
    Calculate offensive/hate speech scores for predictions.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
        soft: If True, return raw probabilities. If False, return binary predictions.

    Returns:
        List[Dict[str, np.ndarray]]: List of dictionaries for each dataset containing:
            - hate: Hate speech score (1=not hate, 0=hate)
            - normal: Normal speech score (1=normal, 0=not normal)
            - offensive: Offensive speech score (1=not offensive, 0=offensive)
    """
    print('-'*80)
    print(f"Computing Offensive/Hate scores with {MODEL_PATHS['offense_hate']}")
    model, tokenizer = load_classifiers(MODEL_PATHS['offense_hate'])
    scores = []
    scores_per_input = []

    for df_idx, df in enumerate(dfs):
        results = []
        preds = df["Prediction"].tolist()
        
        for batch in batchify(preds, args["batch_size"], description=f"{df_idx+1}/{len(dfs)}"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
            
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, -1).cpu().numpy()
                results.extend(probabilities if soft else (probabilities > 0.5).astype(int))
                
        results = np.array(results)

        hate = 1 - statistics.fmean(results[:, 0])
        normal = statistics.fmean(results[:, 1])
        offensive = 1 - statistics.fmean(results[:, 2])
        
        score = statistics.fmean([hate, normal, offensive])
        score_per_input = []
        for i in range(len(results)):
            score_per_input.append(statistics.fmean(
                [1 - results[i, 0], results[i, 1], 1 - results[i, 2]]))
        
        scores.append(score)
        scores_per_input.append(score_per_input)
        cleanup_resources()
        
    return scores, scores_per_input


def compute_argument_score(dfs: List[DataFrame], args: Dict[str, Any], soft: bool = False) -> np.ndarray:
    """
    Calculate counter-argument scores for predictions using a model finetuned on "ThinkCERCA/counterargument_hugging".

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
            - threshold: Classification threshold for binary predictions
        soft: If True, return raw probabilities. If False, return binary predictions.

    Returns:
        np.ndarray: List of counter-argument scores for each dataset (1=good argument, 0=poor argument).
    """
    model, tokenizer = load_classifiers(MODEL_PATHS['argument'], tf=True)

    # TODO add new model to the pipeline _ chkla/roberta-argument

    print('-'*80)
    print(f'Calculating Counter-Argument score with {MODEL_PATHS["argument"]}')
    scores = []
    scores_per_input = []
    
    for df_idx, df in enumerate(dfs):
        results = []
        preds = df["Prediction"].tolist()
        
        for batch in batchify(preds, args["batch_size"], description=f'({df_idx+1}/{len(dfs)})'):
            inputs = tokenizer(batch, return_tensors="tf", padding=True)
            logits = model(**inputs).logits
            probabilities = tf.nn.softmax(logits, -1)[:, 1].cpu().numpy()
            result = probabilities if soft else (probabilities > args["threshold"]).astype(int)
            results.extend([(1 - item).item() for item in result])
        
        scores_per_input.append(results)
        scores.append(1 - statistics.fmean(results))
        cleanup_resources()
        
    return np.array(scores), scores_per_input


def compute_topicRelevance_score(dfs: List[DataFrame], args: Dict[str, Any], infos: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """
    Calculate topic relevance scores for predictions using a classifier finetuned on "cardiffnlp-tweet-topic-21-multi".

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
        infos: Dictionary containing metadata about test sets and models.

    Returns:
        List[Dict[str, np.ndarray]]: List of dictionaries for each dataset containing:
            - f1: Micro-averaged F1 score
            - roc_auc: Micro-averaged ROC AUC score
            - accuracy: Classification accuracy
    """        
    f1_all_models = []
    ac_all_models = []
    results_named_all_models = []
    scores = []
    result_per_input = []
    for M in MODEL_PATHS['topic_relevance']:
        print('-'*80)
        print(f"Computing Topic Relevance scores with {M}")
        model, tokenizer = load_classifiers(M, train=False)

        id2label = model.config.id2label
        label2id = {v: k for k, v in id2label.items()}
        
        model.eval()
        max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 512

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        f1_ = []
        ac_ = []
        results_all_df = []
        for df_idx, df in enumerate(dfs):
            # Handle different possible column names
            target_col = next((col for col in ['Target', 'Labels', 'labels']
                            if col in df.columns), None)

            if target_col is None:
                print(f"Warning: No target column found in dataset {df_idx+1}. Using default labels.")
                # Provide default scores when target information is missing
                scores.append({
                    'f1': np.array(0.5),
                    'accuracy': np.array(0.5)
                })
                continue

            df[target_col] = df[target_col].apply(lambda x: '<WOMEN>' if x == 'sexist' else x)
            df["labels"] = df[target_col].apply(lambda x: [1 if t in x else 0 for t in label2id.keys()])
            results = []
            preds = df["Prediction"].tolist()
            labels = df["labels"].tolist()
            
            with torch.no_grad():
                for batch in batchify(preds, args["batch_size"], description=f"{df_idx+1}/{len(dfs)}"):
                    inputs = tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt"
                        ).to(model.device)
                    
                    outputs = model(**inputs)
                    probs = torch.sigmoid(outputs.logits)
                    batch_preds = (probs >= 0.5).int()
                    results.extend(batch_preds.cpu().numpy())

            results_named = [[id2label[idx] for idx, val in enumerate(pred) if val == 1] for pred in results]
            true_labels_named = [[id2label[idx] for idx, val in enumerate(label) if val == 1] for label in labels]
            
            labels = np.array(labels)
            results = np.array(results)
                
            f1_.append(np.array(f1_score(labels, results, average="micro")))
            ac_.append(np.array(accuracy_score(labels, results)))
            results_all_df.append(results_named)

            cleanup_resources()
        results_named_all_models.append(results_all_df)
        f1_all_models.append(f1_)
        ac_all_models.append(ac_)
        
    for df_idx in range(len(dfs)):
        mean_f1 = np.mean([f1_all_models[model_idx][df_idx]
                          for model_idx in range(len(f1_all_models))])
        mean_ac = np.mean([ac_all_models[model_idx][df_idx]
                          for model_idx in range(len(ac_all_models))])

        scores.append({
            'f1': mean_f1,
            'accuracy': mean_ac
        })
        
        new_labels = []
        for pred_id in range(len(results_named_all_models[0][df_idx])):
            _new_label = [results_named_all_models[model_idx][df_idx][pred_id] for model_idx in range(len(results_named_all_models))]
            new_label = [item for sublist in _new_label for item in sublist]
            new_label = set(new_label)
            new_labels.append(new_label)
            
        result_per_input.append(new_labels)

    return scores, result_per_input


def calculate_ngram_repetition_rate(dfs: List[DataFrame], args: Dict[str, Any]) -> np.ndarray:
    """
    Calculate n-gram repetition rates for predictions.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - n-gram: Size of n-grams to analyze (default: 4)

    Returns:
        np.ndarray: List of repetition rates for each dataset (1=high repetition, 0=low repetition).
    """
    print('-'*80)
    print('Calculating Repetition Rate of predictions')

    rates = []
    n = args.get("n-gram", 4)

    for df_idx, df in enumerate(dfs):
        print(f"Processing: {df_idx+1}/{len(dfs)}")
        text = ' '.join(df["Prediction"])
        tokens = re.findall(r'\b\w+\b', text)
        ngrams_generated = list(ngrams(tokens, n))
        ngrams_unique = len(set(ngrams_generated))
        repetition_rate = 1 - (ngrams_unique / len(ngrams_generated))
        rates.append(1 - repetition_rate)

    return np.array(rates)


def compute_response_length(dfs: List[DataFrame]) -> np.ndarray:
    """
    Calculate average length of predictions and compare to the humen response.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
            - threshold: Classification threshold for binary predictions

    Returns:
        np.ndarray: List of counter-argument scores for each dataset (1=good argument, 0=poor argument).
    """
    print('-'*80)
    print(f'Calculating Prediction Length')
    scores = []
    scores_per_input = []

    for df_idx, df in enumerate(dfs):
        preds = df["Prediction"].tolist()
        results = [len(pred) for pred in preds]
        
        scores_per_input.append(results)
        scores.append(statistics.mean(results))
        cleanup_resources()

    return np.array(scores), scores_per_input


def compute_g_score(eval_: Dict[str, Any], sexism_test: bool) -> float:
    """
    Compute geometric mean of evaluation metrics to get an overall quality score.
    
    Args:
        eval_: Dictionary containing evaluation metrics including:
            - toxicity_score: Score indicating toxicity level (0-1, lower is better)
            - CoLA_score: Grammatical acceptability score (0-1, higher is better)  
            - hate: Hate speech detection score (0-1, lower is better)
            - offensive: Offensive speech detection score (0-1, lower is better)
            - Context_sim_score: Semantic similarity to hate speech context (0-1, higher is better)
            - counter_argument_score: Quality of counter-argument (0-1, higher is better)
            - repetition_rate: Rate of n-gram repetition (0-1, lower is better)
            - f1: Topic relevance F1 score (0-1, higher is better)
            - Label_Cosine_Sim: Semantic similarity to ground truth (0-1, higher is better, excluded for sexism test)
        sexism_test: Boolean indicating if evaluating on sexism test set
        
    Returns:
        float: Geometric mean of all applicable metrics, normalized to 0-1 range.
               Higher values indicate better overall quality.
    """
    metrics = [
        "toxicity_score",
        "CoLA_score", 
        "relevance_score",
        "repetition_rate",
    ]

    # Ensure all required metrics are present
    missing_metrics = [m for m in metrics if m not in eval_]
    if missing_metrics:
        raise KeyError(f"Missing required metrics for G-Score calculation: {missing_metrics}")
    if "counter_argument_score" not in eval_:
        raise KeyError(f"Missing counter_argument_score for G-Score calculation")
    
    return eval_['counter_argument_score'] * statistics.fmean([eval_[metric] for metric in metrics])


def aggregate_toxicity_scores(dfs: List[DataFrame], args: Dict[str, Any], model_paths: List[str]) -> np.ndarray:
    """
    Calculate average toxicity scores across multiple models for predictions.

    Args:
        dfs: List of dataframes containing predictions to evaluate.
        args: Dictionary containing evaluation parameters including:
            - batch_size: Size of batches for processing
        model_paths: List of paths to toxicity models to use for scoring.

    Returns:
        np.ndarray: Array of average toxicity scores for each dataset.
    """
    tox_score = []
    tox_score_per_input = []
    for model_path in model_paths:
        tox_score_temp, tox_score_per_input_temp = compute_toxicity_score(dfs, model_path, args)
        tox_score.append(tox_score_temp)
        tox_score_per_input.append(tox_score_per_input_temp)
        
    scores = np.array(tox_score)
    scores_per_input = np.array(tox_score_per_input)
    return np.mean(scores, axis=0), np.mean(scores_per_input, axis=0)


def compute_perspective_api_score(dfs: List[DataFrame], api_key: str, proxy_info: Dict[str, Any] = None) -> List[Dict[str, float]]:
    """
    Calculate Perspective API scores for predictions.
    
    Args:
        dfs: List of dataframes containing predictions to evaluate.
        api_key: Perspective API key.
        proxy_info: Optional proxy configuration dictionary containing:
            - proxy_host: Proxy host address
            - proxy_port: Proxy port number
            
    Returns:
        List[Dict[str, float]]: List of dictionaries for each dataset containing scores for:
            - TOXICITY
            - IDENTITY_ATTACK
            - INSULT
            - PROFANITY
            - THREAT
            - SEXUALLY_EXPLICIT
    """
    print('-'*80)
    print("Computing Perspective API scores")
    
    # Configure HTTP client with optional proxy
    if proxy_info:
        http = httplib2.Http(
            timeout=10,
            proxy_info=httplib2.ProxyInfo(
                proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
                proxy_host=proxy_info['proxy_host'],
                proxy_port=proxy_info['proxy_port']
            ),
            disable_ssl_certificate_validation=False
        )
    else:
        http = httplib2.Http()


    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        http=http
    )

    scores = []
    for df_idx, df in enumerate(dfs):
        dataset_scores = {
            'TOXICITY': [],
            'IDENTITY_ATTACK': [],
            'INSULT': [],
            'PROFANITY': [],
            'THREAT': [],
            'SEXUALLY_EXPLICIT': []
        }
        
        preds = df["Prediction"].tolist()
        print(f"Processing dataset {df_idx+1}/{len(dfs)}")
        
        for i, text in enumerate(preds):
            try:
                analyze_request = {
                    'comment': {'text': text},
                    'requestedAttributes': {
                        'TOXICITY': {},
                        'IDENTITY_ATTACK': {},
                        'INSULT': {},
                        'PROFANITY': {},
                        'THREAT': {},
                        'SEXUALLY_EXPLICIT': {}
                    },
                    'languages': ['en']
                }
                
                response = client.comments().analyze(body=analyze_request).execute()
                
                # Extract scores for each attribute
                for attr in dataset_scores.keys():
                    score = response['attributeScores'][attr]['summaryScore']['value']
                    dataset_scores[attr].append(1 - score)  # Invert scores so higher is better
                if i % 57 == 0:  # After every 57 requests, wait for 60 seconds
                    print('request limit reached, waiting 60s for next batch.')
                    time.sleep(60)
                    
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                # Add neutral scores on error
                for attr in dataset_scores.keys():
                    dataset_scores[attr].append(0.5)
        
        # Calculate mean scores for the dataset
        mean_scores = {
            attr: statistics.fmean(scores) 
            for attr, scores in dataset_scores.items()
        }
        scores.append(mean_scores)
        
    return scores
