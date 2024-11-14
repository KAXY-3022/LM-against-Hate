from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import Dataset


def concat_hs_cs(df, category:bool):
  '''
  Upon receiving the dataframe, we expect the samples to look like:

  Target                |    Hate_Speech              |   Counter_Speech

  <CLASS-1><CLASS-4>    |   'Some Hateful Comments'   |   'A Counter Reply'

  We concatenate them into
  '<|endoftext|><CLASS-1><CLASS-4> Hate-speech: Some Hateful Comments. Counter-speech: A Counter Reply<|endoftext|>'
  '''
  if category:
    df["text"] = '<|endoftext|>' + df["Target"] + "Hate-speech: " + df['Hate_Speech'] + \
        " " + "Counter-speech: " + df["Counter_Speech"] + '<|endoftext|>'
  else:
    df["text"] = '<|endoftext|>' + "Hate-speech: " + df['Hate_Speech'] + \
        " " + "Counter-speech: " + df["Counter_Speech"] + '<|endoftext|>'
  return df


def concat_hs_label(df):
  '''
  Upon receiving the dataframe, we expect the samples to look like:

  Target                |    Hate_Speech              |   Counter_Speech

  <CLASS-1><CLASS-4>    |   'Some Hateful Comments'   |   'A Counter Reply'

  We concatenate them into
  '<|endoftext|><CLASS-1><CLASS-4> Hate-speech: Some Hateful Comments. Counter-speech: A Counter Reply<|endoftext|>'
  '''

  df["text"] = df["Target"] + df['Hate_Speech']

  return df


def tokenize_labels(df, column_name:str='Target'):
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


def data_init(modeltype: str, tokenizer, df_train, df_val, model=None, category:bool=False):
  '''
  DATA PREPROCESSING and Batching
  '''
  # Tokenize demographic labels
  df_train = tokenize_labels(df_train)
  df_val = tokenize_labels(df_val)
  
  
  # define preprocess function for different modeltypes
  if modeltype == 'Causal':
    df_train = concat_hs_cs(df_train, category)
    df_val = concat_hs_cs(df_val, category)
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
      
    # tokenize dataset
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on val dataset",
    )
    
    '''
    def group_texts(examples):
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    # Chunking texts for batching
    block_size = 512
    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        desc="Chunking training texts for batching",
    )
    val_dataset = val_dataset.map(
        group_texts,
        batched=True,
        desc="Chunking validation texts for batching",
    )
    '''
    # prepare tokenizer for data pre-processing
    tokenizer.pad_token = tokenizer.eos_token
    # set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    
  elif modeltype == 'S2S':
    df_train = concat_hs_label(df_train)
    df_val = concat_hs_label(df_val)
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)

    def preprocess_function(examples):
        # get hate speech as input
        if category:
          inputs = [hatespeech for hatespeech in examples['text']]
        else:
          inputs = [hatespeech for hatespeech in examples['Hate_Speech']]
          
        targets = [
            counterspeech for counterspeech in examples['Counter_Speech']]
        # tokenize the dialogues
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=512, truncation=True)
        return model_inputs
      
    # tokenize dataset
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset and concatenating HS-CS input prompt",
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset and concatenating HS-CS input prompt",
    )

    
    # set up data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
  return train_dataset, val_dataset, data_collator


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