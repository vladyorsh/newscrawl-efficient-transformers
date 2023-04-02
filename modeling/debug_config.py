from .config import get_config as get_base_config

def get_config():
  config = get_base_config()
  
  config['short_max_steps'] = 10
  config['short_eval_steps'] = 10
  config['long_max_steps'] = 10
  config['long_eval_steps'] = 10

  config['short_warmup_steps'] = 0
  config['long_warmup_steps'] = 0

  config['eval_steps'] = 10

  config['train_files'] = [
      'data/news-docs.2009.cs.filtered',
      'data/news-docs.2008.en.filtered',
  ]

  config['valid_files'] = [
      'data/news-docs.2008.cs.filtered',
      'data/news-docs.2007.en.filtered',
  ]

  return config
