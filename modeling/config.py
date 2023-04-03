class StaticAttrDict:
  def __init__(self, d):
    self.d = d
  
  def __getattr__(self, name):
    return self.d.get(name, None)

  def __getitem__(self, key):
    return self.d[key]

  def __setitem__(self, key, newvalue):
    self.d[key] = newvalue

def get_config():
  config = {
    'root_dir' : '.',
    'short_subdir' : 'short',
    'long_subdir' : 'long',

    #Main parameters
    'blocks'      : 12,
    'repeats'     : 1,
    'hidden_dim'  : 768,
    'qkv_dim'     : 768,
    'expansion_dim' : 1536, #3072,
    'num_heads'   : 12,
    'block_size'  : 16,

    'mlm_mask_prob' : 0.15,
    
    #Memory control
    'adafactor' : True,
    'mixed_precision' : False,
    'eval_accumulation_steps' : 1,
    'short_max_len' : 128,
    'long_max_len' : 8192, #None,

    #Dropout and epsilon for clamping divisions
    'attention_dropout_rate' : 0.0,
    'hidden_dropout_rate' : 0.0,
    'eps'         : 1e-12,

    #Training params
    'base_lr' : 1e-3,
    'wd' : 0.01,
    'betas' : (0.9, 0.98),

    'short_full_batch_size' : 1536,
    'long_full_batch_size' : 256,

    'short_train_epochs' : 3.0,
    'short_max_steps' : int(8e4),   #If set, overrides epochs
    'short_eval_steps' : None, #If set, overrides maximum amount of evaluation steps
    'long_train_epochs' : 3.0,
    'long_max_steps' : int(2e4),   #If set, overrides epochs
    'long_eval_steps' : None, #If set, overrides maximum amount of evaluation steps
    
    'short_warmup_steps' : int(1e4), #Linear warmup steps
    'long_warmup_steps' : int(2.5e3),

    'eval_steps' : 10000, #Log, save and eval every ... steps

    'save_total_limit' : 20, #Override older checkpoints if there's more than ...

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
      'data/news-docs.2008.cs.filtered',
      'data/news-docs.2009.cs.filtered',
      'data/news-docs.2010.cs.filtered',
      'data/news-docs.2011.cs.filtered',
      'data/news-docs.2012.cs.filtered',
      'data/news-docs.2013.cs.filtered',
      'data/news-docs.2014.cs.filtered',
      'data/news-docs.2015.cs.filtered',
      'data/news-docs.2016.cs.filtered',
      'data/news-docs.2017.cs.filtered',
      'data/news-docs.2018.cs.filtered',
      'data/news-docs.2020.cs.filtered',
      #'data/news-docs.2007.en.filtered',
      #'data/news-docs.2008.en.filtered',
      #'data/news-docs.2009.en.filtered',
      #'data/news-docs.2010.en.filtered',
      #'data/news-docs.2011.en.filtered',
      #'data/news-docs.2012.en.filtered',
      #'data/news-docs.2013.en.filtered',
      #'data/news-docs.2014.en.filtered',
      #'data/news-docs.2015.en.filtered',
      #'data/news-docs.2016.en.filtered',
      #'data/news-docs.2017.en.filtered',
      'data/news-docs.2018.en.filtered',
      'data/news-docs.2020.en.filtered',
    ],
    'valid_files' : [
      'data/news-docs.2019.cs.filtered',
      'data/news-docs.2021.cs.filtered',
      'data/news-docs.2019.en.filtered',
      #'data/news-docs.2021.en.filtered',
    ],
    'test_files' : [],

    'train_processed_path' : 'train.b',
    'valid_processed_path' : 'valid.b',
    'test_processed_path' : 'test.b',
  }

  if config['mixed_precision']:
    config['eps'] = max(config['eps'], 1e-5)

  return StaticAttrDict(config)
