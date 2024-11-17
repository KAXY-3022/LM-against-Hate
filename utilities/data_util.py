from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
from param import system_message


def add_category_tokens(types, tokenizer, model):
    '''
    Add each target demographic as a new token to tokenizer
    <CLASS1> <CLASS2> etc.
    '''
    new_tokens = []
    for target in types:
        new_tokens.append('<' + target + '>')

        # check if the tokens are already in the vocabulary
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

        # add the tokens to the tokenizer vocabulary
        tokenizer.add_tokens(list(new_tokens))

        # add new, random embeddings for the new tokens
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model


def tokenize_labels(df, column_name: str = 'Target'):
  '''
  extract and tokenize target demographic labels in the original dataset and convert them into predefined class tokens
  '''
  df_test = df.copy()

  def tokenize_classes(item):
    '''
    Each class is predefined as a special token in the model's tokenizer as:
    <CLASS-1>
    The trained model will recognize this label as a special 'word'
     '''
    temp = ''
    if not isinstance(item, list):
      raise ValueError('Target information is not stored as list')
    for ele in item:
      temp = temp + '<' + ele.strip() + '>'
    return temp

  # Concatenate all labels into a string of class tokens <CLASS-1><CLASS-2> etc.
  df_test[column_name] = df_test[column_name].map(tokenize_classes)

  return df_test


def create_chat_template(df, tokenizer, datainfo: str):
  chats = []
  for idx, row in df.iterrows():
    chat = [system_message,
            {'role': 'user', 'content': row['text']},
            {'role': 'assistant', 'content': row['Counter_Speech']}]
    chats.append(chat)
  dataset = Dataset.from_dict({"chat": chats})
  dataset = dataset.map(
      lambda x: {"formated_chat": tokenizer.apply_chat_template(
          x["chat"], tokenize=False, add_generation_prompt=False)},
      desc="applying chat_template to " + datainfo)

  return dataset


def prepare_input_causal(df, category: bool):
  if category:
    # if category tokens are active, add category tokens at beginning of text
    df["text"] = df["Target"] + "Hate-speech: " + df['Hate_Speech'] + \
        " " + "Counter-speech: " + df["Counter_Speech"]
  else:
    df["text"] = "Hate-speech: " + df['Hate_Speech'] + \
        " " + "Counter-speech: " + df["Counter_Speech"]
  return df


def prepare_input_noncausal(df, category: bool):
  if category:
    # if category tokens are active, add category tokens at beginning of text
    df["text"] = df["Target"] + df['Hate_Speech']
  else:
    df["text"] = df['Hate_Speech']
  return df


def setup_datacollator(modeltype: str, tokenizer, model):
  if modeltype == 'Causal':
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
  elif modeltype == 'S2S':
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
  return data_collator


def get_tokenized_chat_template(df, category: bool, tokenizer, loginfo: str = ''):

  def preprocess_function(examples):
    return tokenizer(examples["formated_chat"], truncation=True, add_special_tokens=False)

  df = prepare_input_causal(df=df, category=category)
  dataset = create_chat_template(
      df=df,
      tokenizer=tokenizer,
      datainfo=loginfo)
  dataset = dataset.map(
      preprocess_function,
      batched=True,
      desc="tokenzing " + loginfo)
  return dataset


def get_tokenized_dataset_causal(df, tokenizer, loginfo: str = ''):
  dataset = Dataset.from_pandas(df)

  def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, add_special_tokens=True)

  dataset = dataset.map(
      preprocess_function,
      batched=True,
      desc="tokenzing" + loginfo)
  return dataset


def get_tokenized_dataset_s2s(df, tokenizer, loginfo: str = ''):
  dataset = Dataset.from_pandas(df)

  def preprocess_function(examples):
    # get hate speech as input
    inputs = [hatespeech for hatespeech in examples['text']]
    # get counter speech as target
    targets = [counterspeech for counterspeech in examples['Counter_Speech']]
    # tokenize the dialogues
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs

  # tokenize dataset
  dataset = dataset.map(
      preprocess_function,
      batched=True,
      desc="tokenzing" + loginfo)
  return dataset


def data_init(modeltype: str, params, tokenizer, model=None, category: bool = False):
  # Read csv file into dataframe
  print('Loading dataset: ', params['train_dir'])
  df_train = pd.read_csv(params['train_dir'], converters={'Target': pd.eval})

  print('Loading dataset: ', params['val_dir'])
  df_val = pd.read_csv(params['val_dir'], converters={'Target': pd.eval})

  # Tokenize demographic labels
  df_train = tokenize_labels(df_train)
  df_val = tokenize_labels(df_val)

  if tokenizer.chat_template is not None:
    # if model is instruction tuned, preprocess input with chat_template
    train_dataset = get_tokenized_chat_template(
        df=df_train,
        category=category,
        tokenizer=tokenizer,
        loginfo='train_dataset')

    val_dataset = get_tokenized_chat_template(
        df=df_val,
        category=category,
        tokenizer=tokenizer,
        loginfo='val_dataset')

  # if no chat template exists, use custom defined templates
  elif modeltype == 'Causal':
    df_train = prepare_input_causal(df=df_train, category=category)
    df_val = prepare_input_causal(df=df_val, category=category)

    train_dataset = get_tokenized_dataset_causal(
        df=df_train,
        tokenizer=tokenizer,
        loginfo='train_dataset')
    val_dataset = get_tokenized_dataset_causal(
        df=df_val,
        tokenizer=tokenizer,
        loginfo='val_dataset')

  elif modeltype == 'S2S':
    df_train = prepare_input_noncausal(df=df_train, category=category)
    df_val = prepare_input_noncausal(df=df_val, category=category)

    train_dataset = get_tokenized_dataset_s2s(
        df=df_train,
        tokenizer=tokenizer,
        loginfo='train_dataset')
    val_dataset = get_tokenized_dataset_s2s(
        df=df_val,
        tokenizer=tokenizer,
        loginfo='val_dataset')

  data_collator = setup_datacollator(
      modeltype=modeltype,
      tokenizer=tokenizer,
      model=model)

  return train_dataset, val_dataset, data_collator
