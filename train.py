from modeling.debug_config import *
from modeling.data_processing import NewsCrawlDataset, get_tokenizer, train_tokenizer, make_fast_tokenizer, get_lm_collator
from modeling.models import HTransformer1D, HFWrapper
from modeling.trainer import MyTrainer

import os
import math
import argparse
import torch

from transformers import Trainer, TrainingArguments
from transformers.utils import logging

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', help='Train encoder or decoder', required=False, default='encoder')
parser.add_argument('-c','--short', help='Path to short training checkpoint to continue', required=False, default=None)
parser.add_argument('-C','--long', help='Path to long training checkpoint to continue', required=False, default=None)
args = parser.parse_args()

config = get_config()
is_encoder = args.model.lower() == 'encoder'

def extend_with_rootdir(paths):
  return [ os.path.join(config.root_dir, p) for p in paths ]

#*** GET DATA ***

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

#*** GET TOKENIZER ***
#If cannot retrieve, train new

tok_train_files = config.tokenizer_train_files
if tok_train_files:
  tok_train_files = extend_with_rootdir(tok_train_files)
tok_path = extend_with_rootdir([ config.tokenizer_name])[0]

print('Getting tokenizer from', tok_path)
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
  train_tokenizer(tokenizer, tok_train_data, config)
  tokenizer = make_fast_tokenizer(tokenizer)
  tokenizer.save_pretrained(tok_path)

#*** ESTIMATE AVAILABLE MEMORY AND BATCH SIZES ***

min_memory_available = 1000.0
for device in range(torch.cuda.device_count()):
  t = torch.cuda.get_device_properties(device).total_memory
  r = torch.cuda.memory_reserved(device)
  a = torch.cuda.memory_allocated(device)
  t = t / 1024 ** 2 #MB
  
  device = torch.cuda.get_device_name(device)

  print(f'Device: {device}, memory reserved: {r}, memory allocated: {a}, memory total: {t}')
  
  t = t / 1024
  if t < min_memory_available:
    min_memory_available = t

short_batch_size = int(min_memory_available / 1.0) #1GB per 1 sample of length ~256 (e.g. sentence)
long_batch_size  = round(short_batch_size / 20) #Let it be 20x less

#Extend to power of two
short_batch_size = max(2, 2 ** round(math.log2(short_batch_size)))
long_batch_size  = max(2, 2 ** round(math.log2(long_batch_size)))

print(f'Estimated batch sizes: {short_batch_size} and {long_batch_size} for sentence and document level splits respectively')

#*** CREATE MODEL ***

model = HTransformer1D(config, is_encoder, tokenizer.pad_token_id, None)
model = HFWrapper(model)
short_collator = get_lm_collator(
  tokenizer, padding='longest', max_length=config.short_max_len,
  mask_prob=0.0 if not is_encoder else config.mlm_mask_prob
)
long_collator = get_lm_collator(
  tokenizer, padding='longest', max_length=config.long_max_len,
  mask_prob=0.0 if not is_encoder else config.mlm_mask_prob
)

#def mlm_metrics(eval_preds):
#  logits, labels = eval_preds

#  return {}

#*** DEFINE TRAINING ARGUMENTS ***

common_args = {
  'do_train' : True, 'do_eval' : True, 'do_predict' : True if test_dataset else False,
  'evaluation_strategy' : 'steps', 'gradient_accumulation_steps' : config.grad_accumulation_steps,
  'learning_rate' : config.base_lr, 'weight_decay' : config.wd,
  'logging_first_step' : True, 'logging_steps' : config.eval_steps, 'save_steps' : config.eval_steps,
  'save_total_limit' : config.save_total_limit, 'eval_steps' : None,
  'load_best_model_at_end' : True,
  'eval_accumulation_steps' : config.eval_accumulation_steps,
  'overwrite_output_dir' : True,
}

short_training_args = TrainingArguments(
  per_device_train_batch_size=short_batch_size,
  per_device_eval_batch_size=4 * short_batch_size,
  num_train_epochs=config.short_train_epochs, max_steps=config.short_max_steps,
  warmup_steps=config.short_warmup_steps,
  output_dir=os.path.join(config.root_dir, config.short_subdir),
  ** common_args,
)
long_training_args = TrainingArguments(
  per_device_train_batch_size=long_batch_size,
  per_device_eval_batch_size=4 * long_batch_size,
  num_train_epochs=config.long_train_epochs, max_steps=config.long_max_steps,
  warmup_steps=config.long_warmup_steps,
  output_dir=os.path.join(config.root_dir, config.long_subdir),
  ** common_args,
)

#*** SHORT SEQUENCE PRETRAINING ***

TrainerClass = Trainer
if config.short_eval_steps:
  short_training_args.max_eval_steps=config.short_eval_steps
  TrainerClass = MyTrainer

short_trainer = TrainerClass(
  model,
  short_training_args,
  data_collator=short_collator,
  train_dataset=train_dataset,
  eval_dataset =valid_dataset,
)

#Set datasets to sentence-split mode for short-sequence pretraining
train_dataset.doc_split = False
valid_dataset.doc_split = False
if test_dataset:
  test_dataset.doc_split = False

if args.long is None:
  print('*** Commencing short pretraining ***')
  if args.short is None:
    short_trainer.train()
  else:
    short_trainer.train(args.short)
else:
  print('*** Long checkpoint found, skipping short pretraining ***')

#*** LONG SEQUENCE PRETRAINING

TrainerClass = Trainer
if config.long_eval_steps:
  long_training_args.max_eval_steps=config.long_eval_steps
  TrainerClass = MyTrainer

long_trainer = TrainerClass(
  model,
  long_training_args,
  data_collator=long_collator,
  train_dataset=train_dataset,
  eval_dataset =valid_dataset,
)

#Set datasets to doc-split mode for long-sequence pretraining
train_dataset.doc_split = True
valid_dataset.doc_split = True
if test_dataset:
  test_dataset.doc_split = True

print('*** Commencing long pretraining ***')
if args.long is None:
  long_trainer.train()
else:
  long_trainer.train(args.long)