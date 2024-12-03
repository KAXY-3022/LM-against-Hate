import re
import statistics
import torch
import tensorflow as tf
from sentence_transformers import util
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    
from utilities.cleanup import cleanup_resources
from utilities.batch_processing import batchify
from ..utilities.model_loader import load_classifiers, load_sentence_transformer
from ..config.eval_config import MODEL_PATHS
    
def compute_similarity_pipeline(dfs:list, args:dict, model_paths:list[str], task:str) -> list:
    """
    Compute similarity scores using sentence transformers.

    Args:
        dfs: List of dataframes to evaluate.
        args: Dictionary of evaluation parameters.
        model_paths: List of sentence-transformer models to use for evaluation.
        task: String of task type ('label' or 'context').

    Returns:
        List of similarity scores for each dataset.
    """
    
    scores = np.zeros([len(dfs), len(model_paths)])
    # for each model:
    for model_idx, model_name in enumerate(model_paths):
        print('-'*80)
        print(f"Computing {task} Similarity Score with {model_name}")
        model = load_sentence_transformer(model_name, args["device"])

        for df_idx, df in enumerate(dfs):
            inputs = df["Counter_Speech"].tolist() if task == 'label' else df["Hate_Speech"].tolist()
            preds = df["Prediction"].tolist()
            
            _, cosine_average = compute_similarity_score(
                model, preds, inputs)
            scores[df_idx][model_idx] = cosine_average
            cleanup_resources()

    return scores


def compute_similarity_score(model, preds:list, targets:list):
    """
    Compute cosine similarity scores between predictions and targets.

    Args:
        model: Preloaded sentence transformer model.
        preds: List of predictions.
        targets: List of target sentences.

    Returns:
        A tuple of (cosine scores, average cosine score).
    """
    
    preds_embeddings = model.encode(preds, show_progress_bar=True)
    targets_embeddings = model.encode(targets, show_progress_bar=True)

    cosine_score = util.cos_sim(
        preds_embeddings, targets_embeddings).diagonal()
    # for the sake of scoring between 0-1, we normalize cosine similarity (originally -1 - 1)
    normalized_cosine_score = (cosine_score + 1)/2
    return normalized_cosine_score, statistics.fmean(cosine_score)



def compute_toxicity_score(dfs, model_path, args, soft=False) -> list:
    """
    Calculate toxicity scores for predictions using a pre-trained model.
    This model is trained for toxicity classification task.
    The dataset used for training is the merge of the English parts of the three datasets
    Jigsaw (Jigsaw 2018, Jigsaw 2019, Jigsaw 2020), containing around 2 million examples.
    We split it into two parts and fine-tune a RoBERTa model (RoBERTa: A Robustly Optimized BERT Pretraining Approach) on it.
    The classifiers perform closely on the test set of the first Jigsaw competition,
    reaching the AUC-ROC of 0.98 and F1-score of 0.76.

    Args:
        dfs: List of dataframes with predictions.
        model_path: Path to the toxicity model.
        args: Dictionary of evaluation parameters.
        soft: Whether to return soft probabilities.

    Returns:
        List of toxicity scores for each dataset.
        1: neutral
        0: toxic
    """

    print('-'*80)

    model, tokenizer = load_classifiers(model_path, args['device'])
    scores = []
    for df in dfs:
        results = []
        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for batch in batchify(preds, args["batch_size"], description="Calculating toxicity scores"):
            inputs = tokenizer(batch, return_tensors="pt",
                               padding=True).to(model.device)
            with torch.inference_mode():
                logits = model(**inputs).logits
                
            if soft:
                result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            else:
                result = (logits[:, 1] > 0.5).cpu().numpy()
            results.extend([1 - item for item in result])
        scores.append(statistics.fmean(results))
        cleanup_resources()
    return scores


def compute_cola_score(dfs:list, args:dict, soft:bool=False) -> list:
    """
    Calculate grammatical acceptability (CoLA) scores for predictions.

    Args:
        dfs: List of dataframes containing predictions.
        args: Dictionary of evaluation parameters.
        soft: Whether to return soft probabilities.

    Returns:
        List of CoLA scores for each dataset.
    """
    print('-'*80)

    # initialize model and tokenizer
    model, tokenizer = load_classifiers(MODEL_PATHS['cola'], args['device'])
    scores = []
    for df in dfs:
        results = []
        preds = df["Prediction"].tolist()
        for batch in batchify(preds, args["batch_size"], description="Calculating CoLA acceptability score"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, -1)[:, 0].cpu().numpy()
                results.extend(probabilities if soft else (probabilities > 0.5).astype(int))

        scores.append(1 - statistics.fmean(results))
        cleanup_resources()
    return scores


def compute_Offense_Hate_score(dfs:list, args:dict, soft:bool=False) -> list:
    """
    Calculate offensive/hate speech scores for predictions.

    Args:
        dfs: List of dataframes containing predictions.
        args: Dictionary of evaluation parameters.
        soft: Whether to return soft probabilities.

    Returns:
        List of dictionaries containing (hate_score, normal_score, offensive_score).
        [0:HATE 1:NORMAL 2:OFFENSIVE]
    """
    

    print('-'*80)

    # initialize model and tokenizer
    model, tokenizer = load_classifiers(MODEL_PATHS['offense_hate'], args['device'])

    scores = []
    for df in dfs:
        results = []
        preds = df["Prediction"].tolist()
        for batch in batchify(preds, args["batch_size"], description="Calculating Offensive/Hate scores"):
            inputs = tokenizer(batch,
                               padding=True,
                               truncation=True,
                               return_tensors='pt').to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, -1).cpu().numpy()
                results.extend(probabilities) if soft else (probabilities > 0.5).astype(int)
        results = np.array(results)
        hate_score = 1 - statistics.fmean(results[:, 0])
        normal_score = statistics.fmean(results[:, 1])
        offensive_score = 1 - statistics.fmean(results[:, 2])
        
        # return as dictionary
        metrics = {'hate': hate_score,
                   'normal': normal_score,
                   'offensive': offensive_score}
        scores.append(metrics)
        cleanup_resources()
    return scores


def compute_argument_score(dfs:list, args:dict, soft:bool=False) -> list:
    """
    Calculate counter-argument scores for predictions.
    Classifier is finetuned on "ThinkCERCA/counterargument_hugging"

    Args:
        dfs: List of dataframes containing predictions.
        args: Dictionary of evaluation parameters.
        soft: Whether to return soft probabilities.

    Returns:
        List of counter-argument scores for each dataset.
    """
    # initialize model and tokenizer
    model, tokenizer = load_classifiers(MODEL_PATHS['argument'], args['device'], tf=True)

    # TODO add new model to the pipeline _ chkla/roberta-argument

    print('-'*80)

    scores = []

    for df in dfs:
        results = []
        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for batch in batchify(preds, args["batch_size"], description='Calculating counter-argument score'):
            inputs = tokenizer(batch, return_tensors="tf", padding=True)
            logits = model(**inputs).logits
            probabilities = tf.nn.softmax(logits, -1)[:, 1].cpu().numpy()
            result = probabilities if soft else (probabilities > args["threshold"]).astype(int)
            results.extend([1 - item for item in result])
        scores.append(1 - statistics.fmean(results))
        cleanup_resources()
    return scores




def compute_topicRelevance_score(dfs:list, args:dict, infos:dict) -> list:
    """
    Calculate topic relevance scores for predictions.
    Classifier finetuned on "cardiffnlp-tweet-topic-21-multi"

    Args:
        dfs: List of dataframes containing predictions.
        args: Dictionary of evaluation parameters.
        infos: Metadata about test sets and models.

    Returns:
        List of dictionaries with metrics (F1, ROC AUC, Accuracy) for each dataset.
    """

    # initialize model and tokenizer
    model, tokenizer = load_classifiers(MODEL_PATHS['topic_relevance'], args['device'])

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    print('-'*80)

    scores = []
    sigmoid = torch.nn.Sigmoid()
    
    for df in dfs:
        df["labels"] = df["Target"].apply(lambda x: [1 if t in x else 0 for t in model.config.label2id.keys()])
        results = []
        preds = df["Prediction"].tolist()
        
        for batch in batchify(preds, args["batch_size"], description="Calculating Topic Relevance score"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = sigmoid(torch.Tensor(logits))
            y_pred = (probabilities >= 0.5).astype(int)
            results.extend(y_pred)

        labels = np.array(df["labels"].tolist())

        
        f1 = f1_score(labels, results, average="micro")
        roc_auc = roc_auc_score(labels, results, average="micro", multi_class="ovr")
        accuracy = accuracy_score(labels, results)
        # return as dictionary
        metrics = {'f1': f1,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}

        scores.append(metrics)
        cleanup_resources()

    return scores


def calculate_ngram_repetition_rate(dfs:list, args:dict) -> list:
    """
    Calculate n-gram repetition rates for predictions.

    Args:
        dfs: List of dataframes containing predictions.
        args: Dictionary of evaluation parameters.

    Returns:
        List of repetition rates.
    """
    print('-'*80)
    print('Calculating Repetition Rate of predictions')

    rates = []
    n = args.get("n-gram", 4)

    for df in dfs:
        # concatenate all predictions into one string, connecting with " "
        text = ' '.join(df["Prediction"])
        # Tokenize text
        tokens = re.findall(r'\b\w+\b', text)
        # Generate n-grams
        ngrams_generated = list(ngrams(tokens, n))
        # Count unique n-grams
        ngrams_unique = len(set(ngrams_generated))
        # Calculate repetition rate
        repetition_rate = 1 - (ngrams_unique / len(ngrams_generated))
        rates.append(repetition_rate)

    return rates


def compute_g_score(eval_, infos):
    if infos[0]["Test_Set"] == "Sexism":
        score = [eval_["Toxicity"],
                 eval_["CoLA"],
                 eval_["Hate"],
                 # eval_["Normal"],
                 eval_["Offensive"],
                 # eval_["Label_Cosine_Sim"],
                 # eval_["Label_Dot_Sim"],
                 eval_["Context_Cosine_Sim"],
                 # eval_["Context_Dot_Sim"],
                 eval_["Counter_Argument"],
                 eval_["Repetition_Rate"],
                 eval_["Topic_F1"],
                 ]
    else:
        score = [eval_["Toxicity"],
                 eval_["CoLA"],
                 eval_["Hate"],
                 # eval_["Normal"],
                 eval_["Offensive"],
                 eval_["Label_Cosine_Sim"],
                 # eval_["Label_Dot_Sim"],
                 eval_["Context_Cosine_Sim"],
                 # eval_["Context_Dot_Sim"],
                 eval_["Counter_Argument"],
                 eval_["Repetition_Rate"],
                 eval_["Topic_F1"],
                 ]

    return statistics.fmean(score)



def aggregate_toxicity_scores(dfs, args, model_paths):
    """
    Calculate average toxicity scores across multiple models for predictions.

    Args:
        dfs: List of dataframes with predictions.
        args: Dictionary of evaluation parameters.

    Returns:
        List of average toxicity scores for each dataset.
    """
    aggregated_scores = np.zeros(len(dfs))

    for model_path in model_paths:
        model_scores = compute_toxicity_score(dfs, model_path, args)
        aggregated_scores += np.array(model_scores)

    # Average the scores across the models
    return aggregated_scores / len(model_paths)

