words = open('names.txt', 'r').read().splitlines()
words[:10]
b = {}
for w in words[:1]:
    chs = ['.'] + list(w) + ['.'] #    chs = ['<S>'] + list(w) + ['<E>']#
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

P = (N+1).float() #Smoothing icin +1 ekledik 0 log hesaplamaya kalkmasin diye Laplace smoothing dedi Copilot ilginc... #
##for i in range(27):##
   ##P[i]= P[i] / P[i].sum()##
P = P / P.sum(1, keepdim=True)

n = 0
log_likelihood = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
      n += 1
avg_NLL =-log_likelihood / n
print(avg_NLL)

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




import torch.nn.functional as F
F.one_hot(xs, num_classes=27).float()
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      xs = torch.tensor(itos[x1])
      ys = torch.tensor(itos[x2])
W = torch.randn((27,27), generator=g, requires_grad=True)
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
neg_log_mean = -probs[torch.arange(n), ys].log.mean
W.grad = None loss.backward() W.data += -50 * W.grad
