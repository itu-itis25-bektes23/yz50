words = open('names.txt', 'r').read().splitlines()
words[:10]
b = {}
for w in words[:1]:
    chs = ['.'] + list(w) + ['.'] #    chs = ['<S>'] + list(w) + ['<E>']
#
    for ch1, ch2 in zip(chs, chs[1:]):
      bigram = (ch1, ch2)
      b[bigram] = b.get(bigram, 0) + 1
import torch
N = torch.zeros((27,27) , dtype=torch.int32) #28#
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi['.'] = 26
#stoi['<S>'] = 26#
##stoi['<E>'] = 27##


for w in words:
    chs = ['.'] + list(w) + ['.'] ##chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
      N[stoi[ch1], stoi[ch2]] += 1

itos = {i:s for s,i in stoi.items() }

P = N.float()
##for i in range(27):##
   ##P[i]= P[i] / P[i].sum()##
P = P / P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    out = []
    ix = 26
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 26:
            break
    print(''.join(out))

