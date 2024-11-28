from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
sys.path.append('.')
import torch
from param import model_path



print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(sys.path)
try:
    from flash_attn_2_cuda import some_function
    print("模块加载成功")
except ImportError as e:
    print(f"模块加载失败: {e}")

print('load model')
load_path = model_path.joinpath(
    'Classifiers',
            'cardiffnlp-tweet-topic-21-multi_09,06,2023--21,45')
class_tokenizer = AutoTokenizer.from_pretrained(load_path)
if class_tokenizer.pad_token is None:
    class_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
class_model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            device_map='cuda')

        # set up class mapping
class_mapping = class_model.config.id2label

print('finish')
