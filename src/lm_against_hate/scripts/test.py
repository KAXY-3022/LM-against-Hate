from googleapiclient import discovery
import json
import httplib2.socks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import httplib2
from pathlib import Path

from ..config.config import model_path

proxy_info = httplib2.ProxyInfo(proxy_type=httplib2.socks.PROXY_TYPE_HTTP, proxy_host='127.0.0.1', proxy_port=7890)
http = httplib2.Http(timeout=10, proxy_info=proxy_info, disable_ssl_certificate_validation=False)

json_file_path = "./credentials.json"
with open(json_file_path, "r") as f:
    credentials = json.load(f)
    Perspective_API = credentials['Perspective_API']
    print('loading Perspective API credential: ', Perspective_API)



print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(Path().resolve())


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('Running Perspective API')

API_KEY = Perspective_API

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    http=http
)

analyze_request = {
    'comment': {'text': "I'm not sure what you're trying to say, but I don't think it's fair to call someone a pig just because they're attractive to men."},
    'requestedAttributes': {'TOXICITY': {}, 
                            'IDENTITY_ATTACK': {},
                            'INSULT': {},
                            'PROFANITY': {},
                            'THREAT': {},
                            'SEXUALLY_EXPLICIT': {}},
    "languages": ['en']
}

response = client.comments().analyze(body=analyze_request).execute()
print(json.dumps(response, indent=2))

print('finish')
