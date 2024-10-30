import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from utilities import model_selection, print_gpu_utilization
from pathlib import Path
from eval_util import compute_topicRelevance_score
from tqdm import tqdm
from tqdm.auto import trange
import pandas as pd
import datasets
import os

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


'''
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
'''
'''
model_path = Path().resolve().joinpath('models', 'Classifiers', 'cardiffnlp-tweet-topic-21-multi-09,06,2023--21,45')
print(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

CrowdCounter_dir = Path().resolve().joinpath('data', 'CrowdCounter', 'Final_Dataset_additional_context.json')
print(CrowdCounter_dir)
df_CC = pd.read_json(CrowdCounter_dir)

sigmoid = torch.nn.Sigmoid()
results = []

bs = 15
print(df_CC)

inputs = df_CC["Prediction"].tolist()

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
'''

print_gpu_utilization()
