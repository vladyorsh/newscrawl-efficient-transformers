import torch
import subprocess
import linecache
import base64
import pickle
import gc

from tokenizers import normalizers, pre_tokenizers, models, Tokenizer

from tokenizers.normalizers import NFD #BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.models import Unigram, BPE, WordPiece
from tokenizers.trainers import UnigramTrainer, BpeTrainer, WordPieceTrainer

from transformers import PreTrainedTokenizerFast

class NewsCrawlDataset(torch.utils.data.Dataset):
  def __init__(self, files, doc_split=True):
    self.filenames = files
    
    self.files = []
    self.sizes = []
    self.offsets = []

    self.doc_split = doc_split
    self.sentence_offsets = [] #File -> line -> line offset
    
    for f_idx, filename in enumerate(self.filenames):
      print(f'Indexing {filename}...')
      
      f = open(filename, 'r')
        
      #Add document offsets
      offsets = [ 0 ]
      line = True
      while line: #This won't work with "for line in f"
        line = f.readline()
        if line.strip() == '':
          continue
        offsets.append(f.tell())
        
        #Add sentence offsets
        text = self.line_to_text(line, doc_split=False)
        l_idx = offsets[-2]
        s_idx = [ 0 ]
        c_idx = 0
        for line in text.splitlines(keepends=True):
          c_idx += len(line)
          if line.strip() == '':
            continue
          s_idx.append(c_idx)
        for sent_idx in s_idx[:-1]: #The last is for ''
          self.sentence_offsets.append( (f_idx, l_idx, sent_idx) )

      offsets.pop()
      
      self.sizes.append(len(offsets))
      self.offsets.append(offsets)
      f.close()
      gc.collect()

    self.open()
    print('Dataset created,', len(self), 'lines')

  def open(self):
    self.files = [ open(filename, 'r') for filename in self.filenames ]

  def close(self):
    [ f.close() for f in self.files ]

  def save(self, path):
    with open(path, 'wb') as f:
      pickle.dump((self.filenames, self.sizes, self.offsets, self.sentence_offsets), f)

  @staticmethod
  def load(path, open_files=True):
    d = NewsCrawlDataset([])
    with open(path, 'rb') as f:
      d.filenames, d.sizes, d.offsets, d.sentence_offsets = pickle.load(f)
    if open_files:
      d.open()
    print('Dataset loaded,', len(d), 'lines')
    return d

  def __len__(self):
    size = sum(self.sizes) if self.doc_split else len(self.sentence_offsets)
    return size

  def line_to_text(self, line, doc_split = True):
    idx = 2 if doc_split else 1
    line = line.split('\t')[idx]
    line = base64.b64decode(line).decode('utf-8')
    if doc_split:
      line = line.replace('\n\n', '\n')
    else:
      line = line.strip()
    return line

  def __getitem__(self, idx):
    if self.doc_split:
      file_idx = 0
      for size in self.sizes:
        if idx >= size:
          idx -= size
          file_idx += 1
        else:
          break
      
      #Linecache version is much faster, but also RAM-hungry
      #Just uncomment this and comment everything below
      #line = linecache.getline(self.filenames[file_idx], idx)
      #line = line.split('\t')[2]
      #return base64.b64decode(line).decode('utf-8')
      #line = line.replace('\n\n', '\n') #re.sub('\n+', '\n', line)

      f = self.files[file_idx]
      offset = self.offsets[file_idx][idx]
      f.seek(offset)

      line = f.readline()
      line = self.line_to_text(line, doc_split=True)
      return line
    else:
      file_idx, line_idx, sent_idx = self.sentence_offsets[idx]
      end_idx = None
      if idx != len(self) - 1:
        next_file_idx, next_line_idx, next_sent_idx = self.sentence_offsets[idx+1]
        if next_file_idx == file_idx and next_line_idx == line_idx:
          end_idx = next_sent_idx

      f = self.files[file_idx]
      f.seek(line_idx) #Linecache can be used there as well
      line = f.readline()
      line = self.line_to_text(line, doc_split=False)

      line = line[sent_idx:end_idx]
      #Additionally return offsets so we can tell one document sents from another
      return line#, (file_idx, line_idx, sent_idx)

  def __del__(self):
    for f in self.files:
      if f is not None:
        f.close()

def get_tokenizer(path=None):
  if path is not None:
    return PreTrainedTokenizerFast.from_pretrained(path)
  
  model = WordPiece(unk_token='[UNK]')
  tokenizer = Tokenizer(model)
  tokenizer.normalizer = normalizers.Sequence([ NFD() ])
  tokenizer.pre_tokenizer = pre_tokenizers.Sequence([ BertPreTokenizer() ])

  return tokenizer

def train_tokenizer(tokenizer, iterator, config):
  trainer = WordPieceTrainer(
    vocab_size = config.tokenizer_vocab,
    show_progress = False,
    special_tokens = [ '[PAD]', '[UNK]', '[MASK]', '[SEP]', '[EOS]', '[BOS]' ],
  )

  tokenizer.train_from_iterator(iterator, trainer)
  return tokenizer

def make_fast_tokenizer(tokenizer):
  fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
  special_tokens = {
  'unk_token' : '[UNK]',
  'pad_token' : '[PAD]',
  'mask_token' :'[MASK]',
  'sep_token' : '[SEP]',
  'eos_token' : '[EOS]',
  'bos_token' : '[BOS]',
  }
  fast_tokenizer.add_special_tokens(special_tokens)
  return fast_tokenizer

def refine_decoded_text(s):
  '''
  - Remove spaces before the intra-word "##" tokens
  - Connect back the makeshift "--" em-dash
  '''
  return s.replace(' ##', '').replace('- -', '--')

def get_basic_collator(fast_tokenizer, padding='max_length', max_length=512):
  '''
  padding: { 'longest', 'max_length' }
  max_length: int or None, will truncate if int
  '''

  def collate_fn(batch):
    return fast_tokenizer(
        batch,
        padding=padding,
        max_length=max_length,
        truncation=True,
        return_tensors='pt',
        return_special_tokens_mask=True,
        return_token_type_ids=False,
    )

  return collate_fn

def get_lm_collator(fast_tokenizer, padding='max_length', max_length=512, mask_prob=0.15):
  '''
  padding: { 'longest', 'max_length' }
  max_length: int or None, will truncate if int
  '''

  def collate_fn(batch):
    basic_col = get_basic_collator(fast_tokenizer, padding, max_length)
    inputs = basic_col(batch)

    special_tokens_mask = inputs.special_tokens_mask
    mask_token = fast_tokenizer.mask_token_id
    pad_token  = fast_tokenizer.pad_token_id
    if mask_prob > 1e-5:
      mask = torch.bernoulli(mask_prob * torch.ones(* inputs.input_ids.shape))
      mask = mask * (1 - special_tokens_mask)
      mask = mask.bool()

      #Assign -100 to pad or non-[MASK] tokens
      inputs['labels']   =inputs.input_ids.masked_fill(~mask, -100)
      inputs['input_ids']=inputs.input_ids.masked_fill(mask, mask_token)
    else:
      original_inputs = torch.clone(inputs.input_ids)
      #Causal masking
      input_lengths = inputs.attention_mask.sum(dim=-1) - 1
      #Transform Ids
      inputs['input_ids'][:, input_lengths] = pad_token
      inputs['input_ids'] = inputs.input_ids[:, :-1]
      #Transform attention mask
      inputs['attention_mask'] = inputs.attention_mask[:, 1:]
      #Transform outputs and assign them to inputs dict
      inputs['labels'] = original_inputs[:, 1:]
      #Mask pad tokens with -100
      inputs.labels.masked_fill_(~inputs.attention_mask.bool(), -100)

    inputs.pop('special_tokens_mask')
    return inputs

  return collate_fn