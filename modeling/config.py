class StaticAttrDict:
  def __init__(self, d):
    self.d = d
  
  def __getattr__(self, name):
    return self.d.get(name, None)

  def __getitem__(self, key):
    return self.d[key]

def get_config():
  config = {
    'root_dir' : '.',
    'short_subdir' : 'short',
    'long_subdir' : 'long',

    #Main parameters
    'blocks'      : 12,
    'hidden_dim'  : 768,
    'qkv_dim'     : 768,
    'expansion_dim' : 768,
    'num_heads'   : 12,
    'block_size'  : 16,

    'mlm_mask_prob' : 0.15,
    
    #Memory control
    'eval_accumulation_steps' : 1,
    'short_max_len' : 512,
    'long_max_len' : int(2 ** 14),

    #Dropout and epsilon for clamping divisions
    'attention_dropout_rate' : 0.1,
    'hidden_dropout_rate' : 0.1,
    'eps'         : 1e-12,

    #Training params
    'grad_accumulation_steps' : 8,
    'base_lr' : 5e-5,
    'wd' : 0.0,

    'short_train_epochs' : 3.0,
    'short_max_steps' : int(1e6),   #If set, overrides epochs
    'short_eval_steps' : None, #If set, overrides maximum amount of evaluation steps
    'long_train_epochs' : 3.0,
    'long_max_steps' : int(2e5),   #If set, overrides epochs
    'long_eval_steps' : None, #If set, overrides maximum amount of evaluation steps
    
    'short_warmup_steps' : 0, #Linear warmup steps
    'long_warmup_steps' : 0,

    'eval_steps' : 500, #Log, save and eval every ... steps

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
      'news-docs.2008.cs.filtered',
      'news-docs.2009.cs.filtered',
      'news-docs.2010.cs.filtered',
      'news-docs.2011.cs.filtered',
      'news-docs.2012.cs.filtered',
      'news-docs.2013.cs.filtered',
      'news-docs.2014.cs.filtered',
      'news-docs.2015.cs.filtered',
      'news-docs.2016.cs.filtered',
      'news-docs.2017.cs.filtered',
      'news-docs.2018.cs.filtered',
      'news-docs.2020.cs.filtered',
      'news-docs.2007.en.filtered',
      'news-docs.2008.en.filtered',
      'news-docs.2009.en.filtered',
      'news-docs.2010.en.filtered',
      'news-docs.2011.en.filtered',
      'news-docs.2012.en.filtered',
      'news-docs.2013.en.filtered',
      'news-docs.2014.en.filtered',
      'news-docs.2015.en.filtered',
      'news-docs.2016.en.filtered',
      'news-docs.2017.en.filtered',
      'news-docs.2018.en.filtered',
      'news-docs.2020.en.filtered',
    ],
    'valid_files' : [
      'news-docs.2019.cs.filtered',
      'news-docs.2021.cs.filtered',
      'news-docs.2019.en.filtered',
      'news-docs.2021.en.filtered',
    ],
    'test_files' : [],
  }
  return StaticAttrDict(config)