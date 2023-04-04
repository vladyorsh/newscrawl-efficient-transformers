from .config import get_config as get_base_config

def get_config():
  config = get_base_config()

  config['short_subdir']= 'encoder_short'
  config['long_subdir'] = 'encoder_long'

  config['mlm_mask_prob'] = 0.15
  
  return config
