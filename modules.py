import torch
import torch.nn as nn
import einops
import math

class WordEmbeddings(nn.Module):
  def __init__(self, hidden_dim, vocab_size, padding_idx=0):
    super().__init__()
    self.embs = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)

  def forward(self, x):
    return self.embs(x)

class SineEmbeddings(nn.Module):
  def __init__(self, hidden_dim, eps=1e-12, dropout_rate=0.1):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.layernorm = nn.LayerNorm(hidden_dim, eps=eps)
    self.dropout   = nn.Dropout(dropout_rate)

  def get_sines(self, seq_len, device='cpu'):
    embs = torch.zeros(seq_len, self.hidden_dim, device=device)
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(-1)
    hidden_term = torch.exp(
        (torch.arange(0, self.hidden_dim, 2, dtype=torch.float, device=device)) *
        -(math.log(10000.0) / self.hidden_dim)
    )
    embs[:, 0::2] = torch.sin(position * hidden_term)
    embs[:, 1::2] = torch.cos(position * hidden_term)

    return embs

  def forward(self, x):
    seq_len = x.shape[-2]
    embs = self.get_sines(seq_len, device=x.device)
    embs.unsqueeze(-1)

    rank_diff = len(x.shape) - len(embs.shape)
    for _ in range(rank_diff): embs = embs.unsqueeze(0)

    x = x + embs
    x = self.layernorm(x)
    x = self.dropout(x)

    return x

class Attention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, causal=False, eps=1e-12, dropout_rate=0.1):
    super(Attention, self).__init__()

    self.hidden_dim = hidden_dim
    self.head_dim = qkv_dim // num_heads
    self.num_heads= num_heads
    self.eps = eps
    self.causal = causal

    self.q = nn.Linear(hidden_dim, qkv_dim, bias=False)
    self.k = nn.Linear(hidden_dim, qkv_dim, bias=False)
    self.v = nn.Linear(hidden_dim, qkv_dim, bias=False)

    self.o = nn.Linear(qkv_dim, hidden_dim)

    self.dropout = nn.Dropout(dropout_rate)

  def split_heads(self, x):
    return einops.rearrange(x, '... L (H D) -> ... H L D', H = self.num_heads)
  
  def join_heads(self, x):
    return einops.rearrange(x, '... H L D -> ... L (H D)', H = self.num_heads)

  def forward(self, q, k=None, v=None, query_mask=None, key_mask=None):
    '''
    Expected values:
    - q, k, v: tensors of size "... x SEQ_LEN x HIDDEN_DIM"
    The method works best when there is a strong locality correspondence between queries and keys (e.g. self-attention).
    - query_mask, key_mask: tensors of size "... x SEQ_LEN"
    Mask rank is lower than tensor rank by one (heads and hidden dims are not considered).
    '''
    if k is None:
      k = q
    if v is None:
      v = k
    device = q.device
    if query_mask is None:
      query_mask = torch.ones(size=q.shape[:-1], device=device, dtype=torch.int32)
    if key_mask is None:
      key_mask = query_mask

    q, k, v = self.q(q), self.k(k), self.v(v)
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

    #Unsqueeze masks for heads and for hidden dimension for simplicity (apply the same transforms on masks as on the data)
    query_mask, key_mask = query_mask.unsqueeze(-2), key_mask.unsqueeze(-2)
    query_mask, key_mask = query_mask.unsqueeze(-1), key_mask.unsqueeze(-1)

    q = q * 1 / math.sqrt(self.head_dim)

    logits = torch.einsum('... Q D, ... K D -> ... Q K', q, k)
    mask   = torch.einsum('... Q D, ... K D -> ... Q K', query_mask, key_mask)
    if self.causal:
      mask = torch.tril(mask)
    logits += (1 - mask) * -1e9
    A = nn.functional.softmax(logits, dim=-1)
    A = self.dropout(A)
    out = torch.einsum('... Q K, ... K D -> ... Q D', A, v)
    
    out = self.join_heads(out)
    out = self.o(out)

    return out

class HAttention1D(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, causal=False, block_size = 16, eps=1e-12):
    super(HAttention1D, self).__init__()

    self.hidden_dim = hidden_dim
    self.head_dim = qkv_dim // num_heads
    self.num_heads= num_heads
    self.block_size = block_size #Nr from the paper
    self.eps = eps
    self.causal = causal

    self.q = nn.Linear(hidden_dim, qkv_dim, bias=False)
    self.k = nn.Linear(hidden_dim, qkv_dim, bias=False)
    self.v = nn.Linear(hidden_dim, qkv_dim, bias=False)

    self.o = nn.Linear(qkv_dim, hidden_dim)

  def pad_to_power2(self, x, mask=None):
    seq_len = x.shape[-2]
    new_len = 2 ** math.ceil(math.log2(seq_len))
    
    pad_size = new_len - seq_len
    if pad_size:
      x = nn.functional.pad(x, pad=(0, 0, 0, pad_size))
      if mask is not None:
        mask = nn.functional.pad(mask, pad=(0, pad_size))
    
    return x, mask

  def split_heads(self, x):
    return einops.rearrange(x, '... L (H D) -> ... H L D', H = self.num_heads)
  
  def join_heads(self, x):
    return einops.rearrange(x, '... H L D -> ... L (H D)', H = self.num_heads)

  def to_blocks(self, x, scale_size=1):
    #Doesn't work without 'H' for some reason
    return einops.rearrange(x, '... H (L B) D -> ... H L B D', B = self.block_size * scale_size)

  def swap_block_pairs(self, x):
    x = einops.rearrange(x, '... (L P) B D -> ... L P B D', P = 2)
    x = x.flip(dims=(-3,))
    x = einops.rearrange(x, '... L P B D -> ... (L P) B D', P = 2)
    return x

  def coarsen(self, x, mask=None, avg=False):
    x = einops.rearrange(x, '... H (L P) D -> ... H L P D', P = 2)
    if mask is None:
      retval = x.sum(dim=-2) if not avg else x.mean(dim=-2)
      return retval
    
    mask = einops.rearrange(mask, '... H (L P) D -> ... H L P D', P = 2)
    x = x.masked_fill(1 - mask, 0.0)
    
    agg = x.sum(dim=-2)
    numel=mask.sum(dim=-2)
    if avg:
      agg /= numel.clamp(min=1.0)
    agg.masked_fill_(numel == 0, 0.0)

    return agg, mask.any(dim=-2).int()

  def attend(self, q, k, v, query_mask=None, key_mask=None, blocks_swapped=True):
    q = q * 1 / math.sqrt(self.head_dim)

    #Leave only under-diagonal blocks
    if self.causal and blocks_swapped:
      off_diag = lambda x: x[..., 1::2, :, :]
      q, k, v = off_diag(q), off_diag(k), off_diag(v)
      query_mask, key_mask = off_diag(query_mask), off_diag(key_mask)

    logits = torch.einsum('... Q D, ... K D -> ... Q K', q, k)
    mask   = torch.einsum('... Q D, ... K D -> ... Q K', query_mask, key_mask)

    #Apply triangular mask to the on-diagonal blocks
    if self.causal and not blocks_swapped:
      mask = torch.tril(mask)
    
    logits += (1 - mask) * -1e9 #Masking
    logits -= torch.max(logits, dim=-1, keepdims=True).values #Numeric stabilization
    A = torch.exp(logits) #unnormalized attention scores, A_l in paper

    y = torch.einsum('... Q K, ... K D -> ... Q D', A, v)
    A = A.sum(dim=-1)

    #Restore the over-diagonal zero blocks to retain shape
    if self.causal and blocks_swapped:
      y = einops.repeat(y, ' ... L B D -> ... (L P) B D', P = 2).contiguous()
      A = einops.repeat(A, ' ... L B   -> ... (L P) B'  , P = 2).contiguous()
      
      y[..., 0::2, :, :] = 0
      A[..., 0::2, :]    = 0
      
    #Unblocking the output
    y = einops.rearrange(y, '... L B D -> ... (L B) D')
    A = einops.rearrange(A, '... L B   -> ... (L B)')

    return y, A

  def compute_A_y(self, c_q, c_k, c_v, c_q_mask, c_k_mask, query_ar, key_ar, swap_blocks=True):
    c_q, c_q_mask = self.to_blocks(c_q, scale_size=query_ar), self.to_blocks(c_q_mask, scale_size=query_ar)
    c_k, c_k_mask = self.to_blocks(c_k, scale_size=key_ar),   self.to_blocks(c_k_mask, scale_size=key_ar)
    c_v = self.to_blocks(c_v, scale_size=key_ar)
    
    if swap_blocks:
      c_k, c_k_mask = self.swap_block_pairs(c_k), self.swap_block_pairs(c_k_mask)
      c_v = self.swap_block_pairs(c_v)

    y, A = self.attend(c_q, c_k, c_v, c_q_mask, c_k_mask, swap_blocks)
    return y, A


  def forward(self, q, k=None, v=None, query_mask=None, key_mask=None):
    '''
    Expected values:
    - q, k, v: tensors of size "... x SEQ_LEN x HIDDEN_DIM"
    The method works best when there is a strong locality correspondence between queries and keys (e.g. self-attention).
    - query_mask, key_mask: tensors of size "... x SEQ_LEN"
    Mask rank is lower than tensor rank by one (heads and hidden dims are not considered).
    '''
    if k is None:
      k = q
    if v is None:
      v = k
    device = q.device
    if query_mask is None:
      query_mask = torch.ones(size=q.shape[:-1], device=device, dtype=torch.int32)
    if key_mask is None:
      key_mask = torch.ones(size=k.shape[:-1], device=device, dtype=torch.int32)

    q, k, v = self.q(q), self.k(k), self.v(v)
    
    q, query_mask = self.pad_to_power2(q, query_mask)
    k, key_mask   = self.pad_to_power2(k, key_mask)
    v, _          = self.pad_to_power2(v)

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

    #Unsqueeze masks for heads and for hidden dimension for simplicity (apply the same transforms on masks as on the data)
    query_mask, key_mask = query_mask.unsqueeze(-2), key_mask.unsqueeze(-2)
    query_mask, key_mask = query_mask.unsqueeze(-1), key_mask.unsqueeze(-1)

    q_len, k_len = q.shape[-2], k.shape[-2]
    min_len = min(q_len, k_len)

    #In case of different q-k lengths we compute an aspect ratio of a rectangular block
    #Since inputs are pad to power of two, the aspect ratio also will be a power of two (may be negative)
    aspect_ratio = q_len // k_len if q_len > k_len else k_len // q_len
    queries_longer=q_len > k_len

    num_levels = int(math.log2(min_len // self.block_size)) - 1

    #Coarsening
    #Keep the original q,k,v and work with c_ coarsened versions first
    c_q, c_k, c_v, c_q_mask, c_k_mask = q, k, v, query_mask, key_mask
    coarsened_data = [ (c_q, c_k, c_v, c_q_mask, c_k_mask) ]
    for _ in range(num_levels):
      c_q, c_q_mask = self.coarsen(c_q, c_q_mask, avg=True)
      c_k, c_k_mask = self.coarsen(c_k, c_k_mask, avg=True)
      c_v           = self.coarsen(c_v, avg=False)

      coarsened_data.append( (c_q, c_k, c_v, c_q_mask, c_k_mask) )
    
    #Compute Y_l and A_l for l >= 1 (off-diagonal)
    level_output = []
    
    query_ar, key_ar = (aspect_ratio, 1) if queries_longer else (1, aspect_ratio)
    for c_q, c_k, c_v, c_q_mask, c_k_mask in reversed(coarsened_data):
      y, A = self.compute_A_y(c_q, c_k, c_v, c_q_mask, c_k_mask, query_ar, key_ar)
      level_output.append( (y, A) )

    #Additionally compute on-diagonal blocks for l=0
    y_last, A_last = self.compute_A_y(q, k, v, query_mask, key_mask, query_ar, key_ar, swap_blocks=False)

    #Interpolation
    if self.causal:

      #Produce masks to handle coarsened inputs
      #Only one of 2^level should contribute to output

      uncoarsened_mask = torch.arange(q_len, device=device)
      submasks = [ uncoarsened_mask, uncoarsened_mask ]
      mask = uncoarsened_mask

      for level in range(num_levels):
        mask = einops.rearrange(mask, '(L P) -> L P', P = 2)
        mask = mask.max(dim = -1).values
        expanded_mask = einops.repeat(mask, 'L -> (L P)', P = 2 ** (level + 1))
        submasks.append(expanded_mask)

      submasks = torch.stack(submasks, dim = 0)
      mask = submasks > uncoarsened_mask.unsqueeze(0)

      #BS x H x SEQ_C x D

      rank_diff = len(A_last.shape) - len(mask.shape) + 1 #Number of dims to unsqueeze from left, including extra stack dim
      [ mask.unsqueeze_(0) for _ in range(rank_diff) ]
      mask = einops.rearrange(mask, 'U ... D S -> D ... U S')

      Y_mask = mask.unsqueeze(-1) #Unsqueeze for hidden
      A_mask = mask

      y, A = None, None
      cat  = lambda add, acc: torch.cat([ add.unsqueeze(0), acc ], dim=0)

      for y_level, A_level in level_output:
        if y is None:
          y, A = y_level.unsqueeze(0), A_level.unsqueeze(0)
        else:
          y = einops.repeat(y, '... L D -> ... (L P) D', P = 2)
          A = einops.repeat(A, '... L   -> ... (L P)'  , P = 2)

          y, A = cat(y_level, y), cat(A_level, A)
      
      y, A = cat(y_last, y), cat(A_last, A)

      y, A = y.masked_fill(Y_mask, 0), A.masked_fill(A_mask, 0)
      y, A = y.sum(dim=0), A.sum(dim=0)
    else:
      y, A = 0.0, 0.0
      for y_l, A_l in level_output:
        if torch.is_tensor(y):
          y = einops.repeat(y, '... L D -> ... (L 2) D')
          A = einops.repeat(A, '... L -> ... (L 2)')

        y = y_l + y
        A = A_l + A

      y += y_last
      A += A_last

    out = y / (A + self.eps).unsqueeze(-1)
    out = self.join_heads(out[..., :q_len, :])
    out = self.o(out)

    return out