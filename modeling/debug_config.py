from .config import get_config as get_base_config

def get_config():
  config = get_base_config()

  #config['blocks'] = 1

  config['short_subdir'] = 'debug_short'
  config['long_subdir']  = 'debug_long'

  config['short_max_steps'] = 1000
  config['short_eval_steps'] = 1000
  config['long_max_steps'] = 500
  config['long_eval_steps'] = 500

  config['short_warmup_steps'] = 500
  config['long_warmup_steps'] = 250

  config['eval_steps'] = 250

  config['train_files'] = [
      'data/news-docs.2009.cs.filtered',
  ]

  config['valid_files'] = [
      'data/news-docs.2008.cs.filtered',
  ]

  config['train_processed_path'] = ''
  config['valid_processed_path'] = ''

  return config
