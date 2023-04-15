class StaticAttrDict:
  def __init__(self, d):
    self.d = d
  
  def __getattr__(self, name):
    if name == 'd':
      return self.d
    return self.d.get(name, None)

  def __getitem__(self, key):
    return self.d[key]

  def __setitem__(self, key, newvalue):
    self.d[key] = newvalue

def get_config():
  config = {
    'root_dir' : '.',
    'short_subdir' : 'encoder_short',
    'long_subdir' : 'encoder_long',

    #Main parameters
    'blocks'      : 12,
    'repeats'     : 1,
    'hidden_dim'  : 768,
    'qkv_dim'     : 768,
    'expansion_dim' : 1536, #3072,
    'num_heads'   : 12,

    'mlm_mask_prob' : 0.15,
    
    #Memory control
    'adafactor' : False,
    'mixed_precision' : True,
    'eval_accumulation_steps' : 1,
    'short_max_len' : 128,
    'long_max_len' : 4096, #None,

    #Dropout and epsilon for clamping divisions
    'attention_dropout_rate' : 0.0,
    'hidden_dropout_rate' : 0.0,
    'eps'         : 1e-12,

    #Training params
    'short_base_lr' : 1e-3,
    'long_base_lr'  : 5e-4,
    'wd' : 0.01,
    'betas' : (0.9, 0.98),

    'short_full_batch_size' : 1536,
    'long_full_batch_size' : 256,

    'short_train_epochs' : 3.0,
    'short_max_steps' : int(6e4),   #If set, overrides epochs
    'short_eval_steps' : int(1.5e3), #If set, overrides maximum amount of evaluation steps
    'long_train_epochs' : 3.0,
    'long_max_steps' : int(1.5e4),   #If set, overrides epochs
    'long_eval_steps' : int(5e2), #If set, overrides maximum amount of evaluation steps
    
    'short_warmup_steps' : int(3e4), #Linear warmup steps
    'long_warmup_steps' : int(7.5e3),

    'eval_steps' : 2000, #Log, save and eval every ... steps

    'save_total_limit' : 30, #Override older checkpoints if there's more than ...

    #Combined model
    'use_embeddings' : 'decoder',

    #Efficient variants parameters
    'Nr'          : 16,
    
    #Tokenization
    'tokenizer_vocab' : 30000,
    'tokenizer_train_files' : None, #Use training data
    'tokenizer_name' : 'h_trans_tok',
    
    #Data
    'train_files' : [
      'data/news-docs.2017.cs.filtered',
      'data/news-docs.2018.cs.filtered',
      'data/news-docs.2020.cs.filtered',
      'data/news-docs.2021.en.filtered',
    ],
    'valid_files' : [
      'data/news-docs.2019.cs.filtered',
      'data/news-docs.2021.cs.filtered',
      'data/news-docs.2020.en.filtered',
    ],
    'test_files' : [],

    'train_processed_path' : '', #'train.b',
    'valid_processed_path' : '', #'valid.b',
    'test_processed_path' : '', #'test.b',
  }

  if config['mixed_precision']:
    config['eps'] = max(config['eps'], 1e-5)

  return StaticAttrDict(config)
