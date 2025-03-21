import pandas as pd
import glob
import statistics

from lm_against_hate.config.config import pred_dir
from lm_against_hate.evaluation.metrics import compute_response_length


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

        results, results_per_input = compute_response_length(dfs)

        for idx, length in enumerate(results):
            model_name_ = infos[idx]['Model_Name'] + infos[idx]['Model_Version']
            standard_deviation = statistics.stdev(results_per_input[idx])
            print(f"Length: {model_name_}: {length}")
            print(f"Deviat: {model_name_}: {standard_deviation}")


if __name__ == "__main__":
  main(datasets=[
      'Base', 
      'Sexism', 
      'Small',
      #'test',
      'human_evaluation'
      ])            # 'Base' 'Sexism' 'Small'
