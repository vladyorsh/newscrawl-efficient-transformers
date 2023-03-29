from modeling.config import *
from modeling.data_processing import NewsCrawlDataset, get_tokenizer, train_tokenizer, make_fast_tokenizer, get_lm_collator

import os
import math
import argparse
import gc

config = get_config()

def extend_with_rootdir(paths):
  if isinstance(paths, list) or isinstance(paths, tuple):
    return [ os.path.join(config.root_dir, p) for p in paths ]
  else:
    return os.path.join(config.root_dir, paths)

train_path = extend_with_rootdir(config.train_processed_path)


train_dataset = NewsCrawlDataset([], doc_split=False)

for i, path in enumerate(extend_with_rootdir(config.train_files)):
  savep = f'tmp_{i}.b'
  
  if os.path.exists(savep):
    pass
  else:
    split = NewsCrawlDataset(
      [ path ], doc_split=False
    )
    split.save(savep)
    split.close()
    gc.collect()
  
for i, path in enumerate(extend_with_rootdir(config.train_files)):
  savep = f'tmp_{i}.b'
  split = NewsCrawlDataset.load(savep)

  train_dataset.filenames = train_dataset.filenames + split.filenames
  train_dataset.sizes = train_dataset.sizes + split.sizes
  train_dataset.offsets = train_dataset.offsets + split.offsets
  split.sentence_offsets = [ (i, a, b) for _, a, b in split.sentence_offsets ]
  train_dataset.sentence_offsets = train_dataset.sentence_offsets + split.sentence_offsets

  split.close()
  gc.collect()

for p in os.listdir('.'):
  if 'tmp_' in p:
    os.remove(p)

train_dataset.save(train_path)
print(len(train_dataset))

del train_dataset
gc.collect()

print('Valid dataset')
valid_path = extend_with_rootdir(config.valid_processed_path)
valid_dataset = NewsCrawlDataset(
  extend_with_rootdir(config.valid_files), doc_split=False
)
valid_dataset.save(valid_path)

print('Test dataset')
test_path = config.test_processed_path
if config.test_files:
  test_dataset = NewsCrawlDataset(
    extend_with_rootdir(config.test_files), doc_split=False
  )
  test_dataset.save(test_path)