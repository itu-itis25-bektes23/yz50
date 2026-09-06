words = open('names.txt', 'r').read().splitlines()
words[:10]
b = {}
for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
      bigram = (ch1, ch2)
      b[bigram] = b.get(bigram, 0) + 1
import torch
N = torch.zeros((28,28) , dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
      N[stoi[ch1], stoi[ch2]] += 1

itos = {i:s for s,i in stoi.items() }
import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
plt.imshow(N,itos[i] + itos[j])
for i in 28
    N[i,j].item()
    for j in 28
        plt.text(j, i, itos[i] + itos[j])
cmap='Blues'
plt.axis('off')
