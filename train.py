from code.config import *
from code.data_processing import NewsCrawlDataset, get_tokenizer, train_tokenizer, make_fast_tokenizer
from code.models import HTransformer1D

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', help='Train encoder or decoder', required=False, default='encoder')
args = parser.parse_args()

config = get_config()

def extend_with_rootdir(paths):
  return [ os.path.join(config.root_dir, p) for p in paths ]

#Get the datasets first
print('Parsing train dataset')
train_dataset = NewsCrawlDataset(
    extend_with_rootdir(config.train_files), doc_split=True
)
print('Parsing valid dataset')
valid_dataset = NewsCrawlDataset(
    extend_with_rootdir(config.valid_files), doc_split=True
)
test_dataset = None
if config.test_files:
  print('Parsing test dataset')
  test_dataset = NewsCrawlDataset(
      extend_with_rootdir(config.test_files), doc_split=True
  )

#Get tokenizer, if cannot get -- train new
tok_train_files = config.tokenizer_train_files
if tok_train_files:
  tok_train_files = extend_with_rootdir(tok_train_files)
tok_path = extend_with_rootdir([ config.tokenizer_name])[0]

print('Getting optimizer from', tok_path)
try:
  tokenizer = get_tokenizer(tok_path)
except:
  tokenizer = get_tokenizer()
  print('No trained tokenizer found, prepairing for training')
  tok_train_data = train_dataset
  if tok_train_files:
    print('Parsing tokenizer train dataset')
    tok_train_data = NewsCrawlDataset(
        extend_with_rootdir(config.tokenizer_train_files), doc_split=True
    )
  train_tokenizer(tokenizer, tok_train_data)
  tokenizer = make_fast_tokenizer(tokenizer)
  tokenizer.save_pretrained(tok_path)

#Instantiate a model for training
model = HTransformer1D(config, parser.model.lower() == 'encoder', tokenizer.pad_token_id, None)

