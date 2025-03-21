import torch
from pathlib import Path
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import statistics
from lm_against_hate.evaluation.metrics import load_classifiers
from lm_against_hate.utilities.batch_processing import batchify
from lm_against_hate.utilities.cleanup import cleanup_resources
import tensorflow as tf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(Path().resolve())

df_test = pd.read_csv('data/CONAN/CONAN.csv')
df_test = df_test.dropna(subset=['counterSpeech'])
test_data = df_test['counterSpeech'].tolist()
print(len(test_data))

classifier = pipeline('text-classification', model='chkla/roberta-argument', device=device, batch_size=64)
score = 0
for out in tqdm(classifier(test_data)):
    score += 1 if out['label'] == 'ARGUMENT' else 0
avg_score = score / len(test_data)
print(f"Average argument score: {avg_score:.4f}")






model, tokenizer = load_classifiers(
    'ThinkCERCA/counterargument_hugging', tf=True)

print('-'*80)
print(
    f'Calculating Counter-Argument score with ThinkCERCA/counterargument_hugging')
scores = []
results = []
for batch in batchify(test_data, 64):
    inputs = tokenizer(batch, return_tensors="tf", padding=True)
    logits = model(**inputs).logits
    probabilities = tf.identity(tf.nn.softmax(logits, -1)[:, 1]).numpy()
    result = (probabilities > 0.5).astype(int)
    results.extend([1 - item for item in result])

scores.append(1 - statistics.fmean(results))
print(f'counter-argument score: {scores}')
cleanup_resources()


model, tokenizer = load_classifiers(
    "tum-nlp/bert-counterspeech-classifier", tf=True)

print('-'*80)
print(
    f'Calculating Counter-Argument score with tum-nlp/bert-counterspeech-classifier')
scores = []
results = []
for batch in batchify(test_data, 64):
    inputs = tokenizer(batch, return_tensors="tf", padding=True)
    logits = model(**inputs).logits
    probabilities = tf.identity(tf.nn.softmax(logits, -1)[:, 1]).numpy()
    result = (probabilities > 0.5).astype(int)
    results.extend([1 - item for item in result])

scores.append(1 - statistics.fmean(results))
print(f'counter-argument score: {scores}')
cleanup_resources()



classifier = pipeline(model="facebook/bart-large-mnli",
                      device=device, batch_size=64)
score = 0
outs = classifier(
    test_data,
    candidate_labels=["non-counter-argument", "counter-argument"],
)
scores = [out['scores'][out['labels'].index('counter-argument')] for out in outs]
avg_score = sum(scores) / len(scores)
print(f"Average counter-argument score: {avg_score:.4f}")

print('finish')
