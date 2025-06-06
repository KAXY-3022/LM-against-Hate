import pandas as pd
import glob
from pathlib import Path

from lm_against_hate.config.eval_config import evaluation_args
from lm_against_hate.config.config import pred_dir
from lm_against_hate.evaluation.pipeline import evaluation_pipeline, save_results
from lm_against_hate.utilities.misc import get_datetime


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
            except SyntaxError as se:
                df = pd.read_csv(f)
            except pd.errors.UndefinedVariableError as uve:
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
                    "Test_Set": dataset,
                    "Prediction_File": file_name,
                    }

            dfs.append(df)
            infos.append(info_)

        results, results_per_input = evaluation_pipeline(dfs, infos, evaluation_args)
        
        print(results_per_input)

        save_name = f'evaluation_{dataset}_{get_datetime("%d,%m,%Y--%H,%M")}.csv'
        save_dir = Path().resolve().joinpath('evaluation_results', dataset, save_name)
        save_results(save_dir, results)
        for idx, df in enumerate(results_per_input):
            new_save_name = f'{idx}_{save_name}'
            new_save_dir = Path().resolve().joinpath(
                'evaluation_results', dataset, new_save_name)
            df.to_csv(new_save_dir)


if __name__ == "__main__":
  main(datasets=[
      #'Base', 
      'Sexism', 
      'Small',
      #'test',
      #'human_evaluation'
      ])            # 'Base' 'Sexism' 'Small'
