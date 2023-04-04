from .config import get_config as get_base_config

def get_config():
  config = get_base_config()

  config['short_subdir']= 'decoder_short'
  config['long_subdir'] = 'decoder_long'

  config['mlm_mask_prob'] = 0.0
  
  return config
