from modeling.config import *
from modeling.debug_config import get_config as debug_config
from modeling.encoder_config import get_config as encoder_config
from modeling.decoder_config import get_config as decoder_config

from modeling.data_processing import NewsCrawlDataset, NewsCrawlDatasetLazy, get_tokenizer, train_tokenizer, make_fast_tokenizer, get_lm_collator
from modeling.models import HTransformer1D, HFWrapper, RefTransformer1D
from modeling.trainer import MyTrainer

import os
import math
import argparse
import torch

from transformers import Trainer, TrainingArguments
from transformers.utils import logging
from transformers.debug_utils import DebugUnderflowOverflow

#logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', help='Train encoder or decoder', required=False, default='encoder')
parser.add_argument('-c','--short', help='Path to short training checkpoint to continue', required=False, default=None)
parser.add_argument('-C','--long', help='Path to long training checkpoint to continue', required=False, default=None)
parser.add_argument('-d', '--debug', help='Debug mode', required=False, default=False)
parser.add_argument('-l', '--lazy', help='Lazy data fetching', required=False, default=False)
args = parser.parse_args()

is_encoder = args.model.lower() == 'encoder'
lazy = args.lazy

config = None
if args.debug:
  config = debug_config()
elif is_encoder:
  config = encoder_config()
else:
  config = decoder_config()

def extend_with_rootdir(paths):
  if isinstance(paths, list) or isinstance(paths, tuple):
    return [ os.path.join(config.root_dir, p) for p in paths ]
  else:
    return os.path.join(config.root_dir, paths)

#*** GET DATA ***
'''
print('Parsing train dataset')
if os.path.exists(config.train_processed_path):
  train_path = extend_with_rootdir(config.train_processed_path)
  train_dataset = NewsCrawlDataset.load(train_path)
else:
  print('Creating train index from scratch')
  train_dataset = NewsCrawlDataset(config.train_files)

print('Parsing valid dataset')
if os.path.exists(config.valid_processed_path):
  valid_path = extend_with_rootdir(config.valid_processed_path)
  valid_dataset = NewsCrawlDataset.load(valid_path)
else:
  print('Creating valid index from scratch')
  valid_dataset = NewsCrawlDataset(config.valid_files)


test_dataset = None
'''

#if config.test_files:
#  print('Parsing test dataset')
#  test_path = extend_with_rootdir(config.test_processed_path)
#  if os.path.exists(test_path):
#    test_dataset = NewsCrawlDataset.load(test_path)

#*** GET TOKENIZER ***
#If cannot retrieve, train new

tok_train_files = config.tokenizer_train_files
if tok_train_files:
  tok_train_files = extend_with_rootdir(tok_train_files)
tok_path = extend_with_rootdir(config.tokenizer_name)

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
device_count = 0
for device in range(torch.cuda.device_count()):
  device_count += 1

  t = torch.cuda.get_device_properties(device).total_memory
  r = torch.cuda.memory_reserved(device)
  a = torch.cuda.memory_allocated(device)
  t = t / 1024 ** 2 #MB
  
  device = torch.cuda.get_device_name(device)

  print(f'Device: {device}, memory reserved: {r}, memory allocated: {a}, memory total: {t}')
  
  t = t / 1024
  if t < min_memory_available:
    min_memory_available = t

scaling = 1.0 if not config.mixed_precision else 1.5

#Extend to power of two
short_batch_size = None
if min_memory_available > 42:
  short_batch_size = 128
elif min_memory_available > 36:
  short_batch_size = 96
elif min_memory_available > 30:
  short_batch_size = 64
elif min_memory_available > 22:
  short_batch_size = 48
else:
  short_batch_size = max(2, int(min_memory_available * 512 / config.short_max_len / scaling))

long_batch_size  = max(2, int(short_batch_size * config.short_max_len / config.long_max_len))

short_accum_steps = round(config.short_full_batch_size // short_batch_size / device_count)
long_accum_steps  = round(config.long_full_batch_size // long_batch_size / device_count)

print(f'Estimated batch sizes: {short_batch_size} and {long_batch_size} for sentence and document level splits respectively')

#*** CREATE MODEL ***

model = HTransformer1D(config, is_encoder, tokenizer.pad_token_id, None)
model = HFWrapper(model)

#model = RefTransformer1D(config)


if args.debug and torch.cuda.device_count() == 1:
  debug_overflow = DebugUnderflowOverflow(model)

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
  'do_train' : True, 'do_eval' : True, 'do_predict' : False,
  'evaluation_strategy' : 'steps',
  #'gradient_accumulation_steps' : config.grad_accumulation_steps,
  'learning_rate' : config.base_lr, 'weight_decay' : config.wd,
  'logging_first_step' : True, 'logging_steps' : config.eval_steps, 'save_steps' : config.eval_steps,
  'save_total_limit' : config.save_total_limit, 'eval_steps' : None,
  'load_best_model_at_end' : True,
  'overwrite_output_dir' : True,
  'fp16' : config.mixed_precision,
  'optim' : 'adafactor' if config.adafactor else 'adam',
   'adam_beta1' : config.betas[0], 'adam_beta2' : config.betas[1],
  'eval_accumulation_steps' : config.eval_accumulation_steps,
  'data_seed' : 42,
}

short_training_args = TrainingArguments(
  per_device_train_batch_size=short_batch_size,
  per_device_eval_batch_size=short_batch_size,
  num_train_epochs=config.short_train_epochs, max_steps=config.short_max_steps,
  warmup_steps=config.short_warmup_steps,
  output_dir=os.path.join(config.root_dir, config.short_subdir),
  gradient_accumulation_steps=short_accum_steps,
  ** common_args,
)

long_training_args = TrainingArguments(
  per_device_train_batch_size=long_batch_size,
  per_device_eval_batch_size=long_batch_size,
  num_train_epochs=config.long_train_epochs, max_steps=config.long_max_steps,
  warmup_steps=config.long_warmup_steps,
  output_dir=os.path.join(config.root_dir, config.long_subdir),
  gradient_accumulation_steps=long_accum_steps,
)

print(f'Using {short_training_args.n_gpu} and {long_training_args.n_gpu} GPUs for short and long pretraining.')
print(f'Estimated true training batch sizes are {short_accum_steps * device_count * short_batch_size} and {long_accum_steps * device_count * long_batch_size} respectively.')

#*** READ SENTENCE DATASET ***
if lazy:
  train_dataset = NewsCrawlDatasetLazy(config.train_files, doc_split=False)
  valid_dataset = NewsCrawlDatasetLazy(config.valid_files, doc_split=False)
  test_dataset = None
else:
  train_dataset = NewsCrawlDataset(config.train_files, doc_split=False)
  valid_dataset = NewsCrawlDataset(config.valid_files, doc_split=False)
  test_dataset = None

#*** SHORT SEQUENCE PRETRAINING ***

print(f'Output paths:', config.short_subdir, 'and', config.long_subdir)

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

if args.long is None:
  print('*** Commencing short pretraining ***')
  if args.short is None:
    short_trainer.train()
  else:
    short_trainer.train(args.short)
else:
  print('*** Long checkpoint found, skipping short pretraining ***')

#*** READ DOCUMENT DATASET ***
if not lazy:
  train_dataset = NewsCrawlDataset(config.train_files, doc_split=True)
  valid_dataset = NewsCrawlDataset(config.valid_files, doc_split=True)
else:
  train_dataset.doc_split = False
  valid_dataset.doc_split = False

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

print('*** Commencing long pretraining ***')
if args.long is None:
  long_trainer.train()
else:
  long_trainer.train(args.long)
