import pandas as pd
import glob
from pathlib import Path

from ..config.config import evaluation_args, pred_dir
from ..evaluation.pipeline import evaluation_pipeline, save_results
from ..utilities.misc import get_datetime


def main(datasets):
    for dataset in datasets:
        # set path
        pred_path = pred_dir.joinpath(dataset)
        # read names of all prediction files
        files = glob.glob(str(pred_path) + "/*.csv")

        # Read csv file into dataframe
        dfs = []
        infos = []

        for f in files:
            print('Reading predictions: ', pred_path, f)
            try:
                df = pd.read_csv(f, converters={'Target': pd.eval})
            except pd.errors.UndefinedVariableError:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                print('Empty data in file: ', f)
                continue 
            
            df = df.drop("Unnamed: 0", axis=1)
            file_name = f.replace(str(pred_path), "")
            attr = file_name.replace(".csv", "").split("_")

            df.loc[df["Prediction"].isna(), "Prediction"] = "None"

            info_ = {"Model_Name": attr[0],
                    "Model_Version": attr[1],
                    "Test_Set": attr[2],
                    "Prediction_File": file_name,
                    }

            dfs.append(df)
            infos.append(info_)

        results = evaluation_pipeline(dfs, infos, evaluation_args)

        save_name = f'evaluation_{dataset}_{get_datetime("%d,%m,%Y--%H,%M")}.csv'
        save_dir = Path().resolve().joinpath('evaluation','results', dataset, save_name)
        save_results(save_dir, results)


if __name__ == "__main__":
  main(datasets=['Base', 'Sexism', 'Small'])            # 'Base' 'Sexism' 'Small'
