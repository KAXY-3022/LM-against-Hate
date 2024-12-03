gpt2_medium = {
    "model_type": "Causal",
    "model_name": "gpt2-medium_16,05,2023-21,09",
    "category": False
}

gpt2_medium_category = {
    "model_type": "Causal",
    "model_name": "gpt2-medium-category_11,06,2023--00,31",
    "category": True
}

gpt2_xl = {
    "model_type": "Causal",
    "model_name": "gpt2-xl_07,11,2024--21,00",
    "category": False
}

bart_large = {
    "model_type": "S2S",
    "model_name": "bart-large_18,05,2023--19,18",
    "category": False
}

Llama_32_1B = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct_16,11,2024--00,22",
    "category": False
}

Llama_32_1B_category = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-1B-Instruct-category_17,11,2024--09,39",
    "category": True
}

Llama_32_3B = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct_26,11,2024--00,57",
    "category": False
}

Llama_32_3B_category = {
    "model_type": "Causal",
    "model_name": "Llama-3.2-3B-Instruct-category_22,11,2024--01,23",
    "category": True
}

# list of models to do inference on
INFERENCE_MODELS = [Llama_32_1B,
                    Llama_32_1B_category,
                    Llama_32_3B,
                    Llama_32_3B_category]
