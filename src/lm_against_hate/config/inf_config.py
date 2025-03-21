GPT2_MEDIUM = {
    "model_type": "Causal",
    "model_name": "gpt2-medium_16,05,2023-21,09",
    "category": False
}

GPT2_MEDIUM_CATEGORY = {
    "model_type": "Causal",
    "model_name": "gpt2-medium-category_11,06,2023--00,31",
    "category": True
}

GPT2_XL = {
    "model_type": "Causal",
    "model_name": "gpt2-xl",
    "category": False
}

BART_LARGE = {
    "model_type": "S2S",
    "model_name": "bart-large_18,05,2023--19,18",
    "category": False
}

LLAMA_32_1B = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct",
    "category": False
}
LLAMA_32_1B_ = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct_againstHate_04,12,2024--08,08",
    "category": False
}
LLAMA_32_1B_CATEGORY = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct_againstHate-category_05,12,2024--07,12",
    "category": True
}
LLAMA_32_1B_CATEGORY_2 = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct_againstHate-category_12,02,2025--12,23",
    "category": True
}
LLAMA_32_3B = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct",
    "category": False
}
LLAMA_32_3B_ = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct_againstHate_15,12,2024--03,00",
    "category": False
}
LLAMA_32_3B_CATEGORY = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct_againstHate-category_06,12,2024--04,09",
    "category": True
}
LLAMA_32_3B_CATEGORY2 = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct_againstHate-category_18,12,2024--04,03",
    "category": True
}
FLAN_T5_XL_CATEGORY = {
    "model_type": "S2S",
    "model_name": 'flan-t5-xl-category_02,12,2024--07,33',
    "category": True
}

tester = { 
          'model_type': 'Causal',
          'model_name': 'Llama-3.2-1B-Instruct_againstHate-category_13,02,2025--08,20',
          'category': True}

# list of models to do inference on
INFERENCE_MODELS = [tester]
