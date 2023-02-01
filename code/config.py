def get_config():
  config = {
    #Main parameters
    'blocks'      : 12,
    'hidden_dim'  : 768,
    'qkv_dim'     : 768,
    'expansion_dim' : 768,
    'num_heads'   : 12,
    'block_size'  : 16,

    #Dropout and epsilon for clamping divisions
    'attention_dropout_rate' : 0.1,
    'hidden_dropout_rate' : 0.1,
    'eps'         : 1e-12,

    #Efficient variants parameters
    'Nr'          : 16,
    
    #Tokenization
    'tokenizer_vocab' : 30000,
    'tokenizer_train_files' : None, #Use training data
    'tokenizer_name' : 'h-trans.json',
    
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
  return config