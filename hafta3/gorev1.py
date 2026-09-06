words = open('names.txt', 'r').read.splitlines()
words[:10]

for w in words[:1]
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1
import torch
a = torch.zeros((3,5) , dtype=torch.int32)
