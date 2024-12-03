from config import device

evaluation_args = {"batch_size": 128,
                   "threshold": 0.95,
                   "n-gram": 4,
                   "device": device}

MODEL_PATHS = {
    "cola": 'textattack/roberta-base-CoLA',
    "offense_hate": "Hate-speech-CNERG/bert-base-uncased-hatexplain",
    "argument": "tum-nlp/bert-counterspeech-classifier",
    "topic_relevance": "tum-nlp/roberta-target-demographic-classifier",
    "toxicity": ["martin-ha/toxic-comment-model",
                 'SkolkovoInstitute/roberta_toxicity_classifier'],
    "context_sim": ['multi-qa-MiniLM-L6-cos-v1',
                    "multi-qa-distilbert-cos-v1",
                    # 'multi-qa-mpnet-base-dot-v1',
                    ],
    "label_sim": ['all-MiniLM-L6-v2',
                  'all-mpnet-base-v2',
                  'LaBSE',
                  ]
}