import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from scipy.special import expit
import warnings

from param import model_path, Causal_params, S2S_params, data_path, system_message
from utilities.utils import save_model, model_selection, load_model, print_gpu_utilization, load_local_model


class Dataloader:
    def __init__(self, params: dict):
        self.params = params
        self.__eval = False
        self.df = {}
        self.ds = DatasetDict()

    def __load_dataframe(self, load_dir):
        """
        Load a dataset from a specified directory and drops Unnamed columns.
        """
        print('Loading data from: ', load_dir)
        df = pd.read_csv(load_dir, converters={'Target': pd.eval})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Loaded data is not a pandas DataFrame.")
        
        return self.tokenize_labels(df)

    def eval(self):
        self.__eval = True

    def train(self):
        self.__eval = False

    def get_status(self) -> bool:
        """
        Returns whether the dataloader is in evaluation mode.
        """
        return self.__eval

    def load_train_data(self):
        self.df['train'] = self.__load_dataframe(self.params['train_dir'])

    def load_val_data(self):
        self.df['val'] = self.__load_dataframe(self.params['val_dir'])

    def load_test_data(self):
        self.df['test'] = self.__load_dataframe(self.params['test_dir'])

    def load_custom_data(self, dataset_name: str, load_dir: str):
        self.df[dataset_name] = self.__load_dataframe(load_dir)

    def tokenize_labels(self, df: pd.DataFrame, column_name: str = 'Target') -> pd.DataFrame:
        """
        Tokenize target demographic labels into predefined class tokens.
        """
        def transform_row(row):
            if not isinstance(row[column_name], list):
                raise ValueError('Target information is not stored as list')
            row[column_name] = "".join(
                [f"<{ele.strip()}>" for ele in row[column_name]])
            return row

        tqdm.pandas(desc="Tokenizing labels")
        return df.progress_apply(transform_row, axis=1)

    def prepare_dataset(self, tokenizer=None):
        """
        Prepares datasets by applying input formatting and tokenization.
        """
        if tokenizer is None:
            warnings.warn(
                "No tokenizer passed. A valid tokenizer is required to prepare the dataset.")

        self._prepare_input()
        self._load_dataset(tokenizer=tokenizer)

        if not self.__eval:
            self._tokenize_dataset(tokenizer=tokenizer)

    def _prepare_input(self):
        pass

    def _load_dataset(self, **kwargs):
        """
        Load datasets into Hugging Face's Dataset format.
        """
        for dataset_name, df in self.df.items():
            self.ds[dataset_name] = Dataset.from_pandas(df)

    def _tokenize_dataset(self, **kwargs):
        pass

    def _load_classifier(self, model_name: str = 'cardiffnlp-tweet-topic-21-multi_09,06,2023--21,45'):
        """
        Load target demographic classification model and its tokenizer.
        """
        self.model_name = model_name
        load_path = model_path.joinpath('Classifiers', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            device_map='cuda')
        self.class_mapping = self.model.config.id2label

    def _generate_labels(self, df: pd.DataFrame):
        """
        Generate target demographic labels for rows without existing labels.
        """
        def transform_row(row):
            # If no labels, generate them
            if len(row['Target']) == 0:
                tokens = self.tokenizer(row['Hate_Speech'],
                                        return_tensors='pt').to('cuda')
                output = self.model(**tokens)
                scores = expit(output[0][0].detach().cpu().numpy())
                row["Target"] = [
                    self.class_mapping[i] for i, score in enumerate(scores) if score >= 0.5
                ]
            return row
        
        tqdm.pandas(desc=f"Generating labels with {self.model_name}")
        return df.progress_apply(transform_row, axis=1)


class S2SDataLoader(Dataloader):
    """
    Dataloader for Sequence-to-Sequence (S2S) models without chat templates.
    """

    def __init__(self, param: dict, tokenizer, model):
        super().__init__(param)
        self.datacollator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model)

    def _prepare_input(self, **kwargs):
        include_category = self.params.get('category', False)

        for dataset_name, df in self.df.items():
            self.df[dataset_name]["text"] = (
                df["Target"] +
                df["Hate_Speech"] if include_category else df["Hate_Speech"]
            )

    def _tokenize_dataset(self, tokenizer):

        def preprocess_function(examples):
            inputs = examples['text']
            targets = examples['Counter_Speech']
            return tokenizer(inputs, text_target=targets, max_length=512, truncation=True)

        self.ds = self.ds.map(
            preprocess_function,
            batched=True,
            desc=f"Tokenizing datasets")

        self.ds = self.ds.select_columns(
            ['input_ids', 'attention_mask', 'labels'])


class CausalDataLoader(Dataloader):
    '''
    Dataloader for Causal models without chat templates
    '''

    def __init__(self, param: dict, tokenizer):
        super().__init__(param)
        print('this is a causal dataloader')
        self.datacollator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)

    def _prepare_input(self, **kwargs):
        eval_mode = self.get_status()
        if eval_mode:
            def construct_text(row, include_category):
                text = f"{row['Target']} " if include_category else ""
                row['text'] = (
                    text + f"Hate-speech: {row['Hate_Speech']} Counter-speech:").strip()
                return row
        else:
            def construct_text(row, include_category):
                text = f"{row['Target']} " if include_category else ""
                row['text'] = (
                    text + f"Hate-speech: {row['Hate_Speech']} Counter-speech: {row['Counter_Speech']}").strip()
                return row
            

        include_category = self.params.get('category', False)

        for dataset_name, df in self.df.items():
            tqdm.pandas(desc="Formatting prompt")
            self.df[dataset_name] = df.progress_apply(
                lambda row: construct_text(row, include_category), axis=1)

    def _tokenize_dataset(self, tokenizer):
        self.ds = self.ds.map(
            lambda x: tokenizer(x["text"], 
                                truncation=True,
                                add_special_tokens=True),
            batched=True,
            desc=f"tokenzing dataset")
        self.ds = self.ds.select_columns(['input_ids', 'attention_mask'])


class CTDataLoader(Dataloader):
    '''
    Dataloader for models with chat templates
    '''

    def __init__(self, param: dict, model, tokenizer):
        if tokenizer.chat_template is None:
            raise ValueError(
                'Selected model has no predefined chat_template.')
            
        super().__init__(param)
        self._setup_datacollator(
            modeltype=self.params['model_type'],
            tokenizer=tokenizer,
            model=model)

    def _setup_datacollator(self, modeltype: str, tokenizer, model):
        if modeltype == 'Causal':
            self.datacollator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False)
        elif modeltype == 'S2S':
            self.datacollator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model)

    def _prepare_input(self):
        include_category = self.params.get("category", False)
        for dataset_name, df in self.df.items():
            tqdm.pandas(desc=f'formatting {dataset_name} dataset')
            df["text"] = df["Target"] + df["Hate_Speech"] if include_category else df["Hate_Speech"]
            
    def _load_dataset(self, tokenizer):
        for dataset_name, df in self.df.items():
            self.ds[dataset_name] = self.__format_chat(
                df=df, tokenizer=tokenizer, dataset_name=dataset_name)

    def __format_chat(self, df: pd.DataFrame, tokenizer, dataset_name: str) -> Dataset:
        """
        Formats a dataset for chat-based input with the required structure for the model.
        """
        eval_mode = self.get_status()

        if eval_mode:
            def construct_chat(row):
                return [system_message, {'role': 'user', 'content': row['text']}]
        else:
            def construct_chat(row):
                return [system_message,
                        {'role': 'user', 'content': row['text']},
                        {'role': 'assistant', 'content': row['Counter_Speech']}]

        tqdm.pandas(desc=f'Formatting chat for {dataset_name}')
        df["chat"] = df.progress_apply(
            lambda row: construct_chat(row), axis=1)

        ds = Dataset.from_dict({"chat": df["chat"].tolist()})

        return ds.map(
            lambda x: {
                "formatted_chat": tokenizer.apply_chat_template(
                    x["chat"], tokenize=False, add_generation_prompt=eval_mode)},
            desc=f"Applying chat template to {dataset_name} dataset")
        
    def _tokenize_dataset(self, tokenizer):
        self.ds = self.ds.map(
            lambda x: tokenizer(
                x["formatted_chat"], truncation=True, add_special_tokens=False),
            batched=True,
            remove_columns=["chat"],
            desc=f"Tokenizing dataset")


if __name__ == '__main__':
    causal_param = Causal_params
    causal_param['model_name'] = "gpt2-medium"
    dataloader = Dataloader(causal_param)

    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.load_custom_data(
        dataset_name='EDOS_sexist', load_dir=data_path.joinpath('EDOS_sexist.csv'))
    dataloader.prepare_dataset()
    print(dataloader.df)
    print(dataloader.ds)

    s2s_param = S2S_params
    s2s_param['model_name'] = "bart-large"
    print('Loading model: ', s2s_param['model_name'])
    tokenizer, model = load_model(
        modeltype=s2s_param['model_type'], params=s2s_param, device='cuda')
    dataloader = S2SDataLoader(
        param=s2s_param, tokenizer=tokenizer, model=model)

    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)
    print(dataloader.df)
    print(dataloader.ds)

    dataloader.eval()
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.load_custom_data(
        dataset_name='EDOS_sexist', load_dir=data_path.joinpath('EDOS_sexist.csv'))
    dataloader.prepare_dataset()
    print(dataloader.df)
    print(dataloader.ds)
    
    
    print('Loading model: ', causal_param['model_name'])
    tokenizer, model = load_model(
        modeltype=causal_param['model_type'], params=causal_param, device='cuda')
    dataloader = CausalDataLoader(
        param=causal_param, tokenizer=tokenizer)

    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)
    print(dataloader.df)
    print(dataloader.ds)

    dataloader.eval()
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.load_custom_data(
        dataset_name='EDOS_sexist', load_dir=data_path.joinpath('EDOS_sexist.csv'))
    dataloader.prepare_dataset()
    print(dataloader.df)
    print(dataloader.ds)

    causal_param['model_name'] = "Llama-3.2-3B-Instruct-category_22,11,2024--01,23"
    print('Loading model: ', causal_param['model_name'])
    tokenizer, model = load_local_model(params=causal_param, device='cuda')
    dataloader = CTDataLoader(
        param=causal_param, model=model, tokenizer=tokenizer)

    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.prepare_dataset(tokenizer=tokenizer)
    print(dataloader.df)
    print(dataloader.ds)

    dataloader.eval()
    dataloader.load_train_data()
    dataloader.load_val_data()
    dataloader.load_test_data()
    dataloader.load_custom_data(
        dataset_name='EDOS_sexist', load_dir=data_path.joinpath('EDOS_sexist.csv'))
    dataloader.prepare_dataset(tokenizer=tokenizer)
    print(dataloader.df)
    print(dataloader.ds)
