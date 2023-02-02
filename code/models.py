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
    config = StaticAttrDict(config)

    self.config = config
    self.is_encoder = is_encoder
    
    self.word_embeddings = word_embeddings
    if self.word_embeddings is None:
      self.word_embeddings = WordEmbeddings(config.hidden_dim, config.tokenizer_vocab, padding_idx)
    self.pos_embeddings  = SineEmbeddings(config.hidden_dim, config.eps, config.hidden_dropout_rate)
    
    self.blocks = []
    for _ in range(config.blocks):
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

  def forward(self, q, k=None, v=None, query_mask=None, key_mask=None, return_hidden=False):
    q = self.word_embeddings(q)
    q = self.pos_embeddings(q)

    for block in self.blocks:
      q = block(q, k, v, query_mask, key_mask)

    if return_hidden:
      return q
    q = self.out(q)
    return q

class CombinedModel(nn.Module):
  def __init__(self, encoder_model, decoder_model, padding_idx=0):
    self.encoder = encoder_model
    self.decoder = decoder_model
    self.config  = encoder_model.config

    for block in self.decoder.blocks: #TODO: Replace with other
      block.update_with_cross_attention(
        Attention(
          self.config.hidden_dim, self.config.qkv_dim, self.config.num_heads,
          causal=False, eps=self.config.eps
        )
      )

      if self.config.use_embeddings == 'decoder':
        self.encoder.word_embeddings = self.decoder.word_embeddings
      elif self.config.use_embeddings == 'encoder':
        self.decoder.word_embeddings = self.encoder.word_embeddings
        self.decoder.out.shared_embeddings = self.encoder.word_embeddings
      else:
        self.encoder.word_embeddings = WordEmbeddings(self.config.hidden_dim, self.config.tokenizer_vocab, padding_idx)
        self.decoder.word_embeddings = self.encoder.word_embeddings
        self.decoder.out.shared_embeddings = self.encoder.word_embeddings

  def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
    encoded = self.encoder(encoder_input, query_mask=encoder_mask, key_mask=encoder_mask, return_hidden=True)
    decoded = self.decoder(decoder_input, encoded, encoded, decoder_mask, encoder_mask)

    return decoded