words = open('names.txt', 'r').read().splitlines()
words[:10]
for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    b = {}
    for ch1, ch2 in zip(chs, chs[1:]):
      bigram = (ch1, ch2)
      b[bigram] = b.get(bigram, 0) + 1
import torch
N = torch.zeros((28,28) , dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

itos = {i:s for s,i in stoi.items() }
