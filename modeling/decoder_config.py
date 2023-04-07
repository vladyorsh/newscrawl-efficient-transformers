from .config import get_config as get_base_config
import time

def get_config():
  config = get_base_config()
  gmt = time.time()

  config['short_subdir']= 'decoder_short_' + str(gmt)
  config['long_subdir'] = 'decoder_long_' + str(gmt)

  config['mlm_mask_prob'] = 0.0
  
  return config
