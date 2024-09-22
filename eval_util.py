
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from collections import Counter
import re
from nltk.util import ngrams
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
from numpy.linalg import norm

from tqdm import tqdm
from tqdm.auto import trange

import statistics
import os

from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification

from utilities import cleanup, detokenize


def evaluation_pipeline(dfs, infos, args):
    eval_results = []

    # Counter-Argument
    CA_scores = compute_argument_type(dfs, args)

    # Toxicity Classification Model

    models = ["martin-ha/toxic-comment-model",
              'SkolkovoInstitute/roberta_toxicity_classifier']
    temp_ = 0
    for model_path in models:
        temp = compute_toxicity(dfs, model_path, args)
        temp_ += np.array(temp)

    toxicity_scores = temp_ / len(models)

    # CoLA
    CoLA_scores = compute_cola_transformers(dfs, args)

    # Hate/Normal/Offensive
    HNO_scores = compute_offensiveness(dfs, args)

    # Similarities
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

    # RR
    repetition_rate = calculate_ngram_repetition_rate(dfs, args)

    # Topic
    topics = compute_topic(dfs, args, infos)

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


def compute_similarity_pipeline(dfs, args, models, task):

    scores = np.zeros([len(dfs), len(models)])

    # for each model:
    for m_idx, modelname in enumerate(models):
        print('-'*80)
        print("Computing %s Similarity Score with %s ." % (task, modelname))

        # initialize model
        model = SentenceTransformer(
            'sentence-transformers/' + modelname).to(args["device"])

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


def compute_toxicity(dfs, model_path, args, soft=False):
    # Toxicity Classification Model
    # ---
    # This model is trained for toxicity classification task.
    # The dataset used for training is the merge of the English parts of the three datasets
    # Jigsaw (Jigsaw 2018, Jigsaw 2019, Jigsaw 2020), containing around 2 million examples.
    # We split it into two parts and fine-tune a RoBERTa model (RoBERTa: A Robustly Optimized BERT Pretraining Approach) on it.
    # The classifiers perform closely on the test set of the first Jigsaw competition,
    # reaching the AUC-ROC of 0.98 and F1-score of 0.76.
    # ---
    # 1: neutral
    # 0: toxic

    print('-'*80)
    print('Calculating toxicity of predictions')

    # load tokenizer and model weights
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path).to(args["device"])

    results_list = []
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


def compute_cola_transformers(dfs, args, soft=False):
    # CoLA
    # the lower the better
    print('-'*80)
    print('Calculating CoLA Acceptability Stats')

    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/roberta-base-CoLA").to(args["device"])
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/roberta-base-CoLA")

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


def compute_offensiveness(dfs, args, soft=False):
    # Offensive/Hate
    # [0:HATE 1:NORMAL 2:OFFENSIVE]

    print('-'*80)
    print('Calculating Offensive/Hate-Speech Classification')

    tokenizer = AutoTokenizer.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain").to(args["device"])

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


def compute_argument_type(dfs, args, soft=False):
    # finetuned on "ThinkCERCA/counterargument_hugging"
    model_path = "gdrive/My Drive/Master_Thesis/models/bert-counter-speech-classifier/22,05,2023--21,12"
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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


def compute_topic(dfs, args, infos):
    # finetuned on "ThinkCERCA/counterargument_hugging"
    model_path = "gdrive/My Drive/Master_Thesis/models/sexism_classifiers/cardiffnlp/tweet-topic-21-multi/09,06,2023--21,45"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path).to(args["device"])

    print('-'*80)
    print('Calculating Topic of predictions')

    def format_multiclass(item):
        return item.split("/")

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
            df["Target_2"] = df["Target_2"].map(format_multiclass_2)
            df["Target"] = df["Target"] + df["Target_2"]

        types = [None]*len(model.config.label2id)
        for label, idx in model.config.label2id.items():
            types[int(idx)] = label

        tuple_head = []
        for index, row in df.iterrows():
            label = []
            tuple_temp = [row['Prediction']]
            for target in types:
                if target in row["Target"]:
                    label.append(1)
                else:
                    label.append(0)
            tuple_temp.append(label)
            tuple_head.append(tuple_temp)

        df = pd.DataFrame(tuple_head, columns=["Prediction", "labels"])

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
