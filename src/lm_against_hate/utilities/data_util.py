from ..utilities.DataLoader import CausalDataLoader, S2SDataLoader, CTDataLoader
from ..config.config import data_path

def dataloader_init(param:dict, tokenizer, model, model_type:str) -> CTDataLoader | CausalDataLoader | S2SDataLoader:
  if tokenizer.chat_template is not None:
    return CTDataLoader(param=param, tokenizer=tokenizer, model=model)
  elif model_type == 'Causal':
    return CausalDataLoader(param=param, tokenizer=tokenizer)
  elif model_type == 'S2S':
    return S2SDataLoader(param=param, tokenizer=tokenizer, model=model)
  else:
    raise Exception('given model type is currently not supported.')

def load_custom_dataset(ds_name: str, dataloader: CTDataLoader | CausalDataLoader | S2SDataLoader):
  '''
  ds_name: select which dataset to load
  model_type: S2S or Causal
  gen_tokenizer: tokenizer from the used generator, used to preprocess and tokenize input
  target_awareness: whether the model is trained with target demographic embeddings
  '''
  if ds_name == 'Base':
    # For base comparison
    file_name = 'CONAN_test.csv'

  elif ds_name == 'Small':
    # For comparison with ChatGPT
    file_name = 'T8-S10.csv'

  elif ds_name == 'Sexism':
    # For final comparison, sexism only test set
    file_name = 'EDOS_sexist.csv'

  load_path = data_path.joinpath(file_name)
  dataloader.load_custom_data(dataset_name=ds_name, load_dir=load_path)
