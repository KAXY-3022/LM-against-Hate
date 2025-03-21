import json
from openai import OpenAI, APIConnectionError
import pandas as pd
import numpy as np
from tqdm import tqdm

from lm_against_hate.config.config import data_path_new

load_dir = data_path_new.joinpath('Generator_train.csv')

print('Loading data from: ', load_dir)
df = pd.read_csv(load_dir, converters={'Target': pd.eval})
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
if 'relevance' not in df:
    df['relevance'] = None
    


# Load credentials
json_file_path = "./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
    print('Loading OpenRouter API key')
    api_key = credentials["Open_Router_KEY"]


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def get_answer(row):
    completion = client.chat.completions.create(
        extra_headers={
            # Optional. Site URL for rankings on openrouter.ai.
            "HTTP-Referer": "<YOUR_SITE_URL>",
            # Optional. Site title for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>",
        },
        extra_body={},
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system",
                     "content": "You are a AI assistant that helps NGO operators to combat online hate speech. In this task, you will be labeling hate speech and counter-narrative pairs as either relevant [1] or irrelevant [0]. A relevant counter-narrative should directly address the hateful messages in the hate speech by e.g. providing counter arguments, debunk misinformation or support victims. Please return for each hate speech and counter-narrative pair only a single number."},
            {"role": "user",
             "content": f"Hate Speech: {row['Hate_Speech']} Counter-Narrative: {row['Counter_Speech']}"},
        ],
        temperature=0.5
    )
    return completion.choices[0].message.content, completion.choices[0].message.reasoning

counter = 0
def transform_row_counter(r):
    global counter
    rel = str(r['relevance']).strip()
    if rel == '0' or rel == '1':
        counter += 1
    return r

tqdm.pandas()
df.progress_apply(transform_row_counter, axis=1)
print(counter)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    rel = str(row['relevance']).strip()
    if rel != '0' and rel != '1':
        try:
            relevance, reasoning = get_answer(row)
        except TypeError as te:
            relevance = te
            reasoning = np.nan
        except APIConnectionError as ace:
            relevance = ace
            reasoning = np.nan
        
        df.loc[index, 'relevance'] = relevance
        df.loc[index, 'reasoning'] = reasoning

df.to_csv(data_path_new.joinpath('Generator_train_.csv'))
