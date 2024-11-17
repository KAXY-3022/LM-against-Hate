import sys
sys.path.append('.')
import torch
import pandas as pd
import glob
from pathlib import Path
from utilities.eval_util import evaluation_pipeline, save_results
from utilities.utils import get_datetime


def main(dataset):
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = {"batch_size": 15,
            "threshold": 0.95,
            "n-gram": 4,
            "device": device}

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

    results = evaluation_pipeline(dfs, infos, args)

    save_name = 'evaluation_' + info_["Test_Set"] + \
        "_" + get_datetime("%d,%m,%Y--%H,%M") + '.csv'
    save_dir = Path().resolve().joinpath('evaluation', save_name)
    save_results(save_dir, results)


if __name__ == "__main__":
  main(dataset='Base')            # 'Base' 'Sexism' 'Small'
