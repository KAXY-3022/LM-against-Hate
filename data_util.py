import numpy as np
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq


def concat_hs_cs(df):
  '''
  Upon receiving the dataframe, we expect the samples to look like:

  Target                |    Hate_Speech              |   Counter_Speech

  <CLASS-1><CLASS-4>    |   'Some Hateful Comments'   |   'A Counter Reply'

  We concatenate them into
  '<|endoftext|><CLASS-1><CLASS-4> Hate-speech: Some Hateful Comments. Counter-speech: A Counter Reply<|endoftext|>'
  '''

  df["text"] = '<|endoftext|>' + df["Target"] + "Hate-speech: " + df['Hate_Speech'] + \
      " " + "Counter-speech: " + df["Counter_Speech"] + '<|endoftext|>'

  return df


def extract_labels(df, targets):
  '''
  extract and tokenize target demographic labels in the original dataset and convert them into predefined class tokens
  '''
  df_test = df.copy()
  types = targets
  
  def split_classes(item):
    return item.split("/")
  
  def split_secondary_classes(item):
    if item != item:
      return []
    return item.upper().split(",")
  
  def select_classes(item):
    for ele in item:
      if ele.strip() in types:
        continue
      else:
        item.remove(ele)
    return item
  
  def tokenize_classes(item):
    '''
    Each class is predefined as a special token in the model's tokenizer as:
    <CLASS-1>
    The trained model will recognize this label as a special 'word'
     '''
    temp = ''
    for ele in item:
      temp = temp + '<' + ele.strip() + '>'
    return temp
  
  # Rename Label [Islamophobia] to [MUSLIMS]
  df_test["Target"] = np.where(
      df_test["Target"] == "Islamophobia", "MUSLIMS", df_test["Target"])
  # Format instances which contain multiple labels - seperate each label with /
  df_test["Target"] = df_test["Target"].map(split_classes)
  # Format secondary labels if exist
  df_test["Target_2"] = df_test["Target_2"].map(split_secondary_classes)
  # Merge two sets of labels
  df_test["Target"] = df_test["Target"] + df_test["Target_2"]
  
  # Remove label that are not interested
  df_test["Target"] = df_test["Target"].map(select_classes)
  df_test["Target"] = df_test["Target"].map(select_classes)
  # Concatenate all labels into a string into class tokens <CLASS-1><CLASS-2> etc.
  df_test["Target"] = df_test["Target"].map(tokenize_classes)
  
  return df_test


def preprocess_data_Causal(df, types):
  '''
  Data
  
  For our training, we have a few corpora to work with.
  1. the CONAN datasets
  2. the QIAN Benchmark dataset
  ---
  For fine-tuning GPT models (Causal Language Modeling),
  we concatenate the hate speech and the counter speech in one string
  with each begins with their prefix:
  
  Hate-speech: "Text" Counter-speech: "Text"
  '''
  return concat_hs_cs(extract_labels(df, types))


def data_init(modeltype: str, tokenizer, dataset, model=None):
  '''
  DATA PREPROCESSING and Batching
  '''
  
  # define preprocess function for different modeltypes
  if modeltype == 'Causal':
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
      
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
        
    # tokenize dataset
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on train dataset",
    )
    
    # Chunking texts for batching
    block_size = 512
    dataset = dataset.map(
        group_texts,
        batched=True,
        desc="Chunking texts for batching",
    )
    
    # prepare tokenizer for data pre-processing
    tokenizer.pad_token = tokenizer.eos_token
    # set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
  elif modeltype == 'S2S':
    def preprocess_function(examples):
        # get hate speech as input
        inputs = [hatespeech for hatespeech in examples['Hate_Speech']]
        targets = [
            counterspeech for counterspeech in examples['Counter_Speech']]
        # tokenize the dialogues
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=512, truncation=True)
        return model_inputs
      
    # tokenize dataset
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on train dataset and concatenating HS-CS input prompt",
    )
    
    # set up data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
  return dataset, data_collator


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