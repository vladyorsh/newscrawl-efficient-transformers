import torch
import torch.nn as nn
from config import get_config
from modules import *

class StaticAttrDict:
  def __init__(self, d):
    self.d = d
  
  def __getattr__(self, name):
    return self.d.get(name, None)

  def __getitem__(self, key):
    return self.d[key]

class HTransformer1D(nn.Module):
  def __init__(self, config, is_encoder, padding_idx=0, word_embeddings=None):
    super().__init__()
    self.config = StaticAttrDict(get_config())
    self.is_encoder = is_encoder
    
    self.word_embeddings = word_embeddings
    if self.word_embeddings is None:
      self.word_embeddings = WordEmbeddings(config.hidden_dim, config.tokenizer_vocab, padding_idx)
    self.pos_embeddings  = SineEmbeddings(config.hidden_dim, config.eps, config.hidden_dropout_rate)
    
    self.blocks = []
    for _ in range(config.num_blocks):
      att = HAttention1D(
        config.hidden_dim, config.qkv_dim, config.num_heads,
        causal=not self.is_encoder, block_size = config.Nr, eps=config.eps
      )

      block = None
      if self.is_encoder:
        block = EncoderBlock(
          att, config.hidden_dim, config.expansion_dim,
          config.hidden_dropout_rate, ffn_act=nn.GELU(), eps=config.eps
        )
      else:
        block = DecoderBlock(
          att, None, config.hidden_dim, config.expansion_dim,
          config.hidden_dropout_rate, ffn_act=nn.GELU(), eps=config.eps
        )
      self.blocks.append(block)

    self.blocks = nn.ModuleList(self.blocks)
    self.out = Output(self.word_embeddings)

  def forward(self, q, k=None, v=None, query_mask=None, key_mask=None):
    q = self.word_embeddings(q)
    q = self.pos_embeddings(q)

    for block in self.blocks:
      q = block(q, k, v, query_mask, key_mask)

    q = self.out(q)
    return q
