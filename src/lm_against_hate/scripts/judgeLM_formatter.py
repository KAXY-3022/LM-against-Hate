from os import walk
import pandas as pd
from pathlib import Path

from lm_against_hate.config.config import pred_dir

def get_prediction_files(prediction_path):
    prediction_types = [#'Base', 
                        'Sexism', 
                        'Small'
                        #'human_evaluation'
                        #'test'
                        ]
    pred_dict = {}
    for _type in prediction_types:
        file_list = []
        new_path = prediction_path.joinpath(_type)
        
        for (dirpath, dirnames, filenames) in walk(new_path):
            for file_ in filenames:
                file = pd.read_csv(new_path.joinpath(file_))
                if _type == 'Sexism':
                    unique_values = ['EDOS']
                    file['Dataset'] = 'EDOS'

                    def format_question_id(row):
                        dataset_name = row['Dataset']
                        row['question_id'] = row['Unnamed: 0']
                        return row
                else:
                    try:
                        unique_values = file['Dataset'].unique().tolist()
                    except KeyError as ke:
                        print(ke)
                        print(file_)
                        print(file)
                        raise ke
                    
                    def format_question_id(row):
                        dataset_name = row['Dataset']
                        index = unique_values.index(dataset_name)
                        row['question_id'] = str(index) + str(row['Row ID'])
                        return row
                    
                file = file.apply(format_question_id, axis=1)
                modelname = file_.replace('.csv', '')
                file['model'] = modelname
                file['decoding_method'] = 'top_p_sampling'
                file['scores'] = {}
                file = file.rename(columns={'Hate_Speech':'question_body', 'Prediction':'text'})
                file_list.append(file)
            break
        
        pred_dict[_type] = file_list
        
    return pred_dict


    
def main():
    pred_dict = get_prediction_files(pred_dir)
    for eval_set, df_list in pred_dict.items():
        if eval_set == 'Sexism' or eval_set == 'human_evaluation':
            for df in df_list:
                prediction_file = eval_set + "_" + \
                    df.loc[0]['model'] + ".jsonl"
                prediction_file = Path().resolve().joinpath(
                    'predictions', 'JudgeLM', prediction_file)
                prediction_df = df[[
                    'question_id',
                    'question_body',
                    'decoding_method',
                    'model',
                    'text',
                    'scores']]
                prediction_df = prediction_df.sort_values(by=['question_id'])
                prediction_df.reset_index().to_json(prediction_file, orient="records", lines=True)
        else:
            for df in df_list:
                reference_file = eval_set + "_reference.jsonl"
                reference_file = Path().resolve().joinpath(
                    'predictions', 'JudgeLM', reference_file)
                reference_df = df[[
                    'question_id',
                    'question_body',
                    'decoding_method',
                    'model',
                    'Counter_Speech',
                    'scores']]
                reference_df = reference_df.rename(columns={'Counter_Speech': 'text'})
                reference_df['model'] = ''
                reference_df = reference_df.sort_values(by=['question_id'])
                reference_df.reset_index().to_json(reference_file, orient="records", lines=True)
                break
            
            for df in df_list:
                prediction_file = eval_set + "_" + df.loc[0]['model'] + ".jsonl"
                prediction_file = Path().resolve().joinpath(
                    'predictions', 'JudgeLM', prediction_file)
                prediction_df = df[[
                    'question_id',
                    'question_body',
                    'decoding_method',
                    'model',
                    'text',
                    'scores']]
                prediction_df = prediction_df.sort_values(by=['question_id'])
                prediction_df.reset_index().to_json(prediction_file, orient="records", lines=True)





if __name__ == '__main__':
    main()