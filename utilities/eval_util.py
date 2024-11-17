import ast
import re
import statistics

import torch
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm.auto import trange
from nltk.util import ngrams
from numpy.linalg import norm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification

from utilities.utils import cleanup, detokenize
from param import categories

def evaluation_pipeline(dfs, infos, args):
    # Set up empty stack for storing results
    eval_results = []

    # 1. Counter-Argument Score
    CA_scores = compute_argument_score(dfs, args)

    # 2. Toxicity Score
    models = ["martin-ha/toxic-comment-model",
              'SkolkovoInstitute/roberta_toxicity_classifier']
    temp_ = 0
    for model_path in models:
        temp = compute_toxicity_score(dfs, model_path, args)
        temp_ += np.array(temp)

    toxicity_scores = temp_ / len(models)

    # 3. CoLA Score
    CoLA_scores = compute_cola_score(dfs, args)

    # 4. Hate/Normal/Offensive Score
    HNO_scores = compute_Offense_Hate_score(dfs, args)

    # 5. Similarity Score
    models = ['all-MiniLM-L6-v2',
              'all-mpnet-base-v2',
              'LaBSE',
              ]

    if infos[0]["Test_Set"] != "Sexism":
        Label_sim_scores = compute_similarity_pipeline(
            dfs, args, models, task='label')

    models = ['multi-qa-MiniLM-L6-cos-v1',
              "multi-qa-distilbert-cos-v1",
              # 'multi-qa-mpnet-base-dot-v1',
              ]

    Context_sim_scores = compute_similarity_pipeline(
        dfs, args, models, task='context')

    # 6. Repetition Rate
    repetition_rate = calculate_ngram_repetition_rate(dfs, args)

    # 7. Topic Relevance Score
    topics = compute_topicRelevance_score(dfs, args, infos)

    # 8. Final Scoring
    for idx, info in enumerate(infos):
        # Prepare data for saving evaluation results
        eval_ = {'Model_Name': info["Model_Name"],
                 'Model_Version': info["Model_Version"],
                 'Test_Set': info["Test_Set"],
                 'Prediction_File': info["Prediction_File"],
                 }

        eval_["Toxicity"] = toxicity_scores[idx]
        eval_["CoLA"] = CoLA_scores[idx]

        eval_["Hate"] = HNO_scores[idx][0]
        # eval_["Normal"] = HNO_scores[idx][1]
        eval_["Offensive"] = HNO_scores[idx][2]

        if infos[0]["Test_Set"] != "Sexism":
            eval_["Label_Cosine_Sim"] = statistics.fmean(
                Label_sim_scores[idx][:])

        eval_["Context_Cosine_Sim"] = statistics.fmean(
            Context_sim_scores[idx][:])

        eval_["Counter_Argument"] = CA_scores[idx]
        eval_["Repetition_Rate"] = 1 - repetition_rate[idx]
        eval_["Topic_F1"] = topics[idx]["f1"]

        eval_["G-Score"] = compute_g_score(eval_, infos=infos)
        eval_results.append(eval_)

    return eval_results


def load_classifier_model(model_name: str, device: str, tf: bool = False):
    '''
    model_name: str = name on model card, optimally, the locally stored model should have the same name,
    device: str = 'cuda' or 'cpu'
    '''

    # initialize model and tokenizer
    modelPath = Path().resolve().joinpath('models', 'Classifiers', model_name)

    if tf:
        try:
            # load local model if exists
            tokenizer = AutoTokenizer.from_pretrained(modelPath)
            model = TFAutoModelForSequenceClassification.from_pretrained(
                modelPath)
        except OSError:
            # load model from source
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name)
            # save model locally for future usage
            model.save_pretrained(modelPath)
            tokenizer.save_pretrained(modelPath)
    else:
        try:
            # load local model if exists
            tokenizer = AutoTokenizer.from_pretrained(modelPath)
            model = AutoModelForSequenceClassification.from_pretrained(
                modelPath).to(device)
        except OSError:
            # load model from source
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name).to(device)
            # save model locally for future usage
            model.save_pretrained(modelPath)
            tokenizer.save_pretrained(modelPath)

    return model, tokenizer


def compute_similarity_pipeline(dfs, args, models, task):
    '''
    Compute similarity scores using sentence transformers
    
    dfs: a list of dataframes to be evaluated
    args: evaluation arguments
    models: selected models to use for evaluation
    task: evaluation task, label similarity or context similarity
    '''
    scores = np.zeros([len(dfs), len(models)])

    # for each model:
    for m_idx, modelname in enumerate(models):
        print('-'*80)
        print("Computing %s Similarity Score with %s ." % (task, modelname))

        # initialize model
        modelPath = str(Path().resolve().joinpath(
            'models', 'Classifiers', 'sentence-transformers', modelname))
        try:
            # load local model if exists
            model = SentenceTransformer(modelPath).to(args["device"])
        except ValueError:
            # load model from source
            model = SentenceTransformer(
                'sentence-transformers/' + modelname).to(args["device"])
            # save model locally for future use
            model.save(modelPath)

        for df_idx, df in enumerate(dfs):

            if task == 'label':
                inputs = df["Counter_Speech"].tolist()
            elif task == 'context':
                inputs = df["Hate_Speech"].tolist()

            preds = df["Prediction"].tolist()
            # list, scalar, list, scalar
            cosine_results, cosine_score = compute_similarity_score(
                model, preds, inputs)

            scores[df_idx][m_idx] = cosine_score

            cleanup()

    return scores


def compute_similarity_score(model, preds, targets):
    preds_embeddings = model.encode(preds, show_progress_bar=True)
    targets_embeddings = model.encode(targets, show_progress_bar=True)

    cosine_score = util.cos_sim(
        preds_embeddings, targets_embeddings).diagonal()
    # for the sake of scoring between 0-1, we normalize cosine similarity (originally -1 - 1)
    cosine_score = (cosine_score + 1)/2
    cosine_average = statistics.fmean(cosine_score)

    return cosine_score, cosine_average


def compute_toxicity_score(dfs, model_path, args, soft=False):
    '''
    Toxicity Classification Model
    ---
    This model is trained for toxicity classification task.
    The dataset used for training is the merge of the English parts of the three datasets
    Jigsaw (Jigsaw 2018, Jigsaw 2019, Jigsaw 2020), containing around 2 million examples.
    We split it into two parts and fine-tune a RoBERTa model (RoBERTa: A Robustly Optimized BERT Pretraining Approach) on it.
    The classifiers perform closely on the test set of the first Jigsaw competition,
    reaching the AUC-ROC of 0.98 and F1-score of 0.76.
    ---
    1: neutral
    0: toxic
    '''

    print('-'*80)
    print('Calculating toxicity of predictions')

    # initialize model and tokenizer
    model, tokenizer = load_classifier_model(
        model_name=model_path, device=args['device'])

    scores = []

    for df in dfs:
        results = []
        bs = args["batch_size"]

        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for i in tqdm(range(0, len(preds), bs)):
            batch = tokenizer(preds[i:i + bs],
                              return_tensors='pt',
                              padding=True).to(model.device)

            with torch.inference_mode():
                logits = model(**batch).logits
            if soft:
                result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            else:
                result = (logits[:, 1] > 0.5).cpu().numpy()
            results.extend([1 - item for item in result])

            average = statistics.fmean(results)

        scores.append(average)
        cleanup()

    return scores


def compute_cola_score(dfs, args, soft=False):
    # CoLA
    # the lower the better
    print('-'*80)
    print('Calculating CoLA Acceptability Stats')

    # initialize model and tokenizer
    model, tokenizer = load_classifier_model(
        model_name='textattack/roberta-base-CoLA', device=args['device'])

    scores = []

    for df in dfs:
        results = []
        bs = args["batch_size"]

        preds = df["Prediction"].tolist()
        for i in trange(0, len(preds), bs):
            batch = [detokenize(t) for t in preds[i: i + bs]]
            inputs = tokenizer(batch,
                               padding=True,
                               truncation=True,
                               return_tensors='pt').to(model.device)
            with torch.no_grad():
                out = torch.softmax(
                    model(**inputs).logits, -1)[:, 0].cpu().numpy()
            if soft:
                results.append(out)
            else:
                results.append((out > 0.5).astype(int))

        results = np.concatenate(results)
        average = 1 - statistics.fmean(results)

        scores.append(average)
        cleanup()

    return scores


def compute_Offense_Hate_score(dfs, args, soft=False):
    # Offensive/Hate
    # [0:HATE 1:NORMAL 2:OFFENSIVE]

    print('-'*80)
    print('Calculating Offensive/Hate-Speech Classification')

    # initialize model and tokenizer
    model, tokenizer = load_classifier_model(
        model_name="Hate-speech-CNERG/bert-base-uncased-hatexplain", device=args['device'])

    scores = []

    for df in dfs:
        results = []
        bs = args["batch_size"]
        preds = df["Prediction"].tolist()
        for i in trange(0, len(preds), bs):
            batch = [detokenize(t) for t in preds[i: i + bs]]
            inputs = tokenizer(batch,
                               padding=True,
                               truncation=True,
                               return_tensors='pt').to(model.device)

            with torch.no_grad():
                out = torch.softmax(
                    model(**inputs).logits, -1).cpu().numpy()
            if soft:
                results.append(out)
            else:
                results.append((out > 0.5).astype(int))

        results = np.concatenate(results)

        hate_score = 1 - statistics.fmean(results[:, 0])
        normal_score = statistics.fmean(results[:, 1])
        offensive_score = 1 - statistics.fmean(results[:, 2])

        # scores saved as a tuple
        score = (hate_score, normal_score, offensive_score)
        scores.append(score)

        cleanup()

    return scores


def compute_argument_score(dfs, args, soft=False):
    '''
    finetuned on "ThinkCERCA/counterargument_hugging"
    '''
    # initialize model and tokenizer
    model, tokenizer = load_classifier_model(
        model_name="tum-nlp/bert-counterspeech-classifier", device=args['device'], tf=True)

    # TODO add new model to the pipeline _ chkla/roberta-argument

    print('-'*80)
    print('Calculating Argument Type of predictions')

    scores = []

    for df in dfs:
        results = []
        bs = args["batch_size"]
        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for i in tqdm(range(0, len(preds), bs)):
            batch = tokenizer(
                preds[i:i + bs],
                return_tensors='tf',
                padding=True)

            with torch.inference_mode():
                logits = model(**batch).logits

            if soft:
                result = tf.nn.softmax(logits, -1)[:, 1].cpu().numpy()
            else:
                result = (logits[:, 1] > args["threshold"]).cpu().numpy()

            results.extend([1 - item for item in result])
            average = 1 - statistics.fmean(results)

        scores.append(average)
        cleanup()

    return scores




def compute_topicRelevance_score(dfs, args, infos):
    '''
    finetuned on "cardiffnlp-tweet-topic-21-multi"
    '''

    # initialize model and tokenizer
    model, tokenizer = load_classifier_model(
        model_name='tum-nlp/roberta-target-demographic-classifier', device=args['device'])

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print('-'*80)
    print('Calculating Topic of predictions')

    def format_multiclass(item):
        if isinstance(item, str):
            return item.split("/")
        else:
            return item

    def format_multiclass_2(item):
        if item != item:
            return []
        return item.upper().split(",")

    scores = []

    for df in dfs:
        if infos[0]["Test_Set"] == "Sexism":
            df["Target"] = np.where(df["labels"] == "sexist", "WOMEN", "other")

        elif infos[0]["Test_Set"] == "Base":
            df["Target"] = np.where(
                df["Target"] == "Islamophobia", "MUSLIMS", df["Target"])
            df["Target"] = df["Target"].map(format_multiclass)

            if 'Target_2' in df.columns:
                df["Target_2"] = df["Target_2"].map(format_multiclass_2)
                df["Target"] = df["Target"] + df["Target_2"]

        types = [None]*len(model.config.label2id)
        for label, idx in model.config.label2id.items():
            types[int(idx)] = label

        def transform_row(r):
            labels = []
            for target in types:
                if target in r['Target']:
                    labels.append(1)
                else:
                    labels.append(0)
            r['labels'] = labels
            return r

        df = df.apply(transform_row, axis=1)

        sigmoid = torch.nn.Sigmoid()

        results = []
        bs = args["batch_size"]
        # prepare the input
        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for i in tqdm(range(0, len(preds), bs)):
            batch = tokenizer(
                preds[i:i + bs],
                return_tensors='pt',
                padding=True).to(model.device)

            with torch.inference_mode():
                logits = model(**batch).logits

            probs = sigmoid(torch.Tensor(logits))
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs.cpu() >= 0.5)] = 1

            results = results + y_pred.tolist()

        labels = df["labels"].tolist()

        f1_micro_average = f1_score(
            y_true=labels, y_pred=results, average='micro')
        roc_auc = roc_auc_score(labels, results, average='micro')
        accuracy = accuracy_score(labels, results)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}

        scores.append(metrics)
        cleanup()

    return scores


def calculate_ngram_repetition_rate(dfs, args):
    print('-'*80)
    print('Calculating Repetition Rate of predictions')

    scores = []
    if hasattr(args, "n-gram"):
        n = args["n-gram"]
    else:
        n = 4

    for df in dfs:
        texts = df["Prediction"]

        # concatenate all predictions into one string, connecting with " "
        text = ' '.join(texts)

        # Tokenize the text
        tokens = re.findall(r'\b\w+\b', text)

        # Generate n-grams
        generated_ngrams = list(ngrams(tokens, n))

        # Count unique n-grams
        unique_ngrams = len(set(generated_ngrams))

        # Total n-grams
        total_ngrams = len(generated_ngrams)

        # Calculate repetition rate
        repetition_rate = 1 - (unique_ngrams / total_ngrams)

        scores.append(repetition_rate)

    return scores


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


def save_results(save_dir, eval):
    df = pd.DataFrame(eval)
    df.to_csv(save_dir)







# Unused test codes

def compute_bertopic_score(dfs, args, infos):
    load_path = Path().resolve().joinpath('models', 'BERTopic_model')
    topic_model = BERTopic.load(load_path)

    le = LabelEncoder()
    le.classes_ = np.load(
        load_path.joinpath('classes.npy'), allow_pickle=True)

    print('-'*80)
    print('Calculating Topic of predictions with BERTopic')

    scores = []

    for df in dfs:
        if infos[0]["Test_Set"] == "Sexism":
            df["Target"] = np.where(df["labels"] == "sexist", "WOMEN", "other")

        results = []
        bs = args["batch_size"]
        # prepare the input
        preds = df["Prediction"].tolist()
        # Batch Evaluation
        for i in tqdm(range(0, len(preds), bs)):
            batch = preds[i:i + bs]

            classes, probs = topic_model.transform(batch)
            y_preds = le.inverse_transform(classes).tolist()

            def eval(string):
                return ast.literal_eval(string)

            # y_preds = list(map(eval, y_preds))
            results = results + y_preds

        labels = df["Target"].tolist()
        count = 0
        for i in range(len(labels)):
            print(labels[i], results[i])
            if results[i] in labels[i]:
                count += 1
        print(count/len(labels))