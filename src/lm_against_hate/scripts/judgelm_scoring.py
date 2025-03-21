import pandas as pd
from pathlib import Path

judgement = Path().resolve().joinpath('evaluation_results', 'JudgeLM')
judgement = judgement.joinpath('Llama-sexism-new copy')

print(f"Judgement content: {judgement}")

df = pd.read_json(judgement, lines=True)

print(df['pred_text'])

def get_score_fast(row):
    row['score'] = row['pred_text'].split(' ')
    return row


def get_score_full(row):
    scores = row['pred_text'].split('\n')[0]
    row['score'] = scores.split(' ')
    return row

get_score = get_score_fast if 'fast' in str(judgement) else get_score_full

df = df.apply(get_score, axis=1)
print(df['score'])
models = []
for key, value in df.loc[1].items():
    if 'model_id' in key:
        print(key)
        models.append(value)
models.pop()
# Split the list of scores into separate columns
scores_df = pd.DataFrame(df["score"].to_list(), columns=models)
scores_df = scores_df.apply(pd.to_numeric, errors='coerce')

# Calculate the average for each model
average_scores = scores_df.mean() / 10

print(average_scores)
