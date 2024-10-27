import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from utilities import model_selection, print_gpu_utilization
from pathlib import Path


print(torch.__version__)
print(torch.cuda.is_available())


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
params = model_selection(modeltype='Causal',modelname='google/flan-t5-large')

model_name = params['model_name'].split('/')[-1]
load_path = Path().resolve().joinpath(
        params['save_dir'],
        params['training_args'].output_dir,
        model_name)

tokenizer = AutoTokenizer.from_pretrained(load_path) 
model = AutoModelForCausalLM.from_pretrained(load_path)

print_gpu_utilization()