from .config import get_config as get_base_config

def get_config():
  config = get_base_config()

  #config['blocks'] = 1
  
  config['short_max_steps'] = 100
  config['short_eval_steps'] = 100
  config['long_max_steps'] = 50
  config['long_eval_steps'] = 50

  config['short_warmup_steps'] = 50
  config['long_warmup_steps'] = 25

  config['eval_steps'] = 10

  config['train_files'] = [
      'data/news-docs.2009.cs.filtered',
  ]

  config['valid_files'] = [
      'data/news-docs.2008.cs.filtered',
  ]

  config['train_processed_path'] = ''
  config['valid_processed_path'] = ''

  return config
