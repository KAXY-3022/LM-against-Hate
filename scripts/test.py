
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from bertopic import BERTopic
import ast
import sys
import pandas as pd
import glob
sys.path.append('.')
import torch
from pathlib import Path
from utilities.utils import model_selection, print_gpu_utilization, load_model, cleanup, load_local_model
from utilities.eval_util import compute_bertopic_score
import numpy


print(torch.__version__)
print(torch.cuda.is_available())


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
    
args = {"batch_size": 15,
        "threshold": 0.95,
        "n-gram": 4,
        "device": device}

dataset = 'test'
# set path
pred_dir = Path().resolve().joinpath('predictions', dataset)

# read names of all prediction files
files = glob.glob(str(pred_dir) + "\*.csv")

# Read csv file into dataframe
dfs = []
infos = []

for f in files:
    print('Reading predictions: ', pred_dir, f)
    try:
        df = pd.read_csv(f, converters={'Target': pd.eval})
    except pd.errors.UndefinedVariableError:
        df = pd.read_csv(f)
    df = df.drop("Unnamed: 0", axis=1)
    file_name = f.replace(str(pred_dir), "")
    attr = file_name.replace(".csv", "").split("_")

    for idx, pred in enumerate(df["Prediction"].tolist()):
        if pred != pred:
            df.loc[idx, "Prediction"] = "None"

    info_ = {"Model_Name": attr[0],
                 "Model_Version": attr[1],
                 "Test_Set": attr[2],
                 "Prediction_File": file_name,
                 }

    dfs.append(df)
    infos.append(info_)


compute_bertopic_score(dfs=dfs, args=args, infos=infos)



print_gpu_utilization()
