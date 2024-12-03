import sys
sys.path.append('.')
import pandas as pd
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from scipy.special import expit
import warnings

from ..config.config import system_message, categories
from ..utilities.model_loader import load_classifiers


class Dataloader:
    def __init__(self, params: dict):
        self.params = params
        self.__eval = False
        self.df = {}
        self.ds = DatasetDict()
        self.include_category = self.params.get("category", False)

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

    def load_classifier(self, classifier_name: str = 'cardiffnlp-tweet-topic-21-multi_09,06,2023--21,45'):
        self.classifier_name = classifier_name
        
        print(f'loading classifier: {classifier_name}')
        self.classifier, self.tokenizerr = load_classifiers(classifier_name)
        self.class_mapping = self.classifier.config.id2label

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
        eval_mode = self.get_status()
        if eval_mode and self.include_category:
            if self.classifier is None:
                warnings.warn('No classifier is loaded, skipping label generation. Provide a classifier model to the dataloader before re-loading the dataset to perform correct label generation if required.')
            else:
                # set up class mapping
                df = self._generate_labels(df=df)
                
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


    def _generate_labels(self, df: pd.DataFrame):
        """
        Generate target demographic labels for rows without existing labels.
        """
        def transform_row(row):
            # If no labels, generate them
            if len(row['Target']) == 0:
                tokens = self.tokenizer(row['Hate_Speech'],
                                        return_tensors='pt').to('cuda')
                output = self.classifier(**tokens)
                scores = expit(output[0][0].detach().cpu().numpy())
                row["Target"] = [
                    self.class_mapping[i] for i, score in enumerate(scores) if score >= 0.5
                ]
            return row
        
        tqdm.pandas(desc=f"Generating labels with {self.classifier_name}")
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
        if self.include_category:
            def construct_text(row):
                row['text'] = f'{row['Target']} {row['Hate_Speech']}'
                return row
        else:
            def construct_text(row):
                row['text'] = row['Hate_Speech']
                return row
        
        for dataset_name, df in self.df.items():
            tqdm.pandas(desc=f"Formatting prompt: {dataset_name}")
            self.df[dataset_name] = df.progress_apply(construct_text, axis=1)

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
        self.datacollator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)

    def _prepare_input(self, **kwargs):
        eval_mode = self.get_status()
        
        if eval_mode and self.include_category:
            def construct_text(row):
                row['text'] = (
                    f"{row['Target']} Hate-speech: {row['Hate_Speech']} Counter-speech:").strip()
                return row
        elif eval_mode and not self.include_category:
            def construct_text(row):
                row['text'] = (f"Hate-speech: {row['Hate_Speech']} Counter-speech:").strip()
                return row
        elif not eval_mode and self.include_category:
            def construct_text(row):
                row['text'] = (f"{row['Target']} Hate-speech: {row['Hate_Speech']} Counter-speech: {row['Counter_Speech']}").strip()
                return row
        elif not eval_mode and not self.include_category:
            def construct_text(row):
                row['text'] = (f"Hate-speech: {row['Hate_Speech']} Counter-speech: {row['Counter_Speech']}").strip()
                return row

        for dataset_name, df in self.df.items():
            tqdm.pandas(desc=f"Formatting prompt: {dataset_name}")
            self.df[dataset_name] = df.progress_apply(construct_text, axis=1)

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
        if self.include_category:
            def construct_text(row):
                row['text'] = f'{row['Target']} {row['Hate_Speech']}'
                return row
        else:
            def construct_text(row):
                row['text'] = row['Hate_Speech']

        for dataset_name, df in self.df.items():
            tqdm.pandas(desc=f"Formatting prompt: {dataset_name}")
            self.df[dataset_name] = df.progress_apply(construct_text, axis=1)
            
     
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

        tqdm.pandas(desc=f'Formatting chat for {dataset_name} dataset')
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

class ClassifierDataLoader(Dataloader):
    '''
    Dataloader for Classifier models
    '''
    def __init__(self, param: dict, tokenizer):
        super().__init__(param)
        self.datacollator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def _prepare_input(self):
        # Format labels for multi-target classification
        def construct_labels(row):
            for cat in categories:
                row[cat] = float(row[cat])
            row["labels"] = [row[c] for c in categories]
            return row
        
        for dataset_name, df in self.df.items():
            counter_columns = categories + ['Counter_Speech']
            df_c = df[counter_columns]
            df_c = df_c[df_c['Counter_Speech'].notna()].rename(columns={
                "Counter_Speech": "text"})

            hate_columns = categories + ['Hate_Speech']
            df_h = df[hate_columns]
            df_h = df_h[df_h['Hate_Speech'].notna()].rename(columns={"Hate_Speech": "text"})

            self.df[dataset_name] = pd.concat([df_h, df_c], ignore_index=True)
            tqdm.pandas(desc=f"Formatting labels for {dataset_name}")
            self.df[dataset_name] = self.df[dataset_name].progress_apply(construct_labels, axis=1)
   
    def _tokenize_dataset(self, tokenizer):
        cols = self.ds['train'].column_names
        cols.remove("labels")
        self.ds = self.ds.map(
            lambda x: tokenizer(
                x["text"], truncation=True, max_length=512, padding=True, return_tensors='pt'),
            batched=True,
            remove_columns=cols)

if __name__ == '__main__':
    causal_dataloader = CausalDataLoader()
    s2s_dataloader = S2SDataLoader()
    ct_dataloader = CTDataLoader()
    