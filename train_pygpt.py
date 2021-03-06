#!/usr/bin/env python3
import math
import torch
import numpy as np
from torch.utils.data import Dataset

class CharDataset(Dataset):
  def __init__(self, data, block_size):
    #chars = sorted(list(set(data)))
    chars = '\x00\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    
    self.stoi = { ch:i for i,ch in enumerate(chars) }
    self.itos = { i:ch for i,ch in enumerate(chars) }
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.data = data
  
  def __len__(self):
    #return math.ceil(len(self.data) / (self.block_size + 1))
    return 1024*128

  def __getitem__(self, idx):
    # we're actually going to "cheat" and pick a spot in the dataset at random
    i = np.random.randint(0, len(self.data) - (self.block_size + 1))
    chunk = self.data[i:i+self.block_size+1]
    dix = [self.stoi[s] for s in chunk]
    x = torch.tensor(dix[:-1], dtype=torch.long)
    y = torch.tensor(dix[1:], dtype=torch.long)
    return x, y


if __name__ == "__main__":
  print("starting")

  block_size = 128
  text = open("/raid.dell2/pygpt/pydata_test.txt").read() #.read(1024*1024*4)
  train_dataset = CharDataset(text, block_size)

  from mingpt.model import GPT, GPTConfig
  mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=8, n_head=8, n_embd=512)
  model = GPT(mconf)

  #model.load_state_dict(torch.load("/raid.dell2/pygpt/model.state"))
  
  from mingpt.trainer import Trainer, TrainerConfig
  tconf = TrainerConfig(max_epochs=200, batch_size=128*2, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20,
                        final_tokens=200*len(train_dataset)*block_size,
                        ckpt_path="/raid.dell2/pygpt/model.state",
                        num_workers=4)
  trainer = Trainer(model, train_dataset, None, tconf)
  trainer.train()

