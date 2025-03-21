import json

from lm_against_hate.config.config import device

json_file_path = "./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
    Perspective_API = credentials['Perspective_API']
    print('loading Perspective API credential: ', Perspective_API)


evaluation_args = {"batch_size": 128,
                   "threshold": 0.95,
                   "n-gram": 4,
                   "device": device,
                   #"perspective_api_key": Perspective_API,
                   # Optional proxy configuration
                   "proxy_info": {
                       "proxy_host": "127.0.0.1",
                       "proxy_port": 7890
                   },
                   }

MODEL_PATHS = {
    "cola": 'textattack/roberta-base-CoLA',
    "offense_hate": "Hate-speech-CNERG/bert-base-uncased-hatexplain",
    "argument": "ThinkCERCA/counterspeech_",
    "topic_relevance":["NLP-LTU\\target_demographic_bertweet-large-sexism-detector", 
                       'cardiffnlp\\target_demographic_tweet-topic-21-multi'],
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