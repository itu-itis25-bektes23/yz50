words = open('names.txt', 'r').read().splitlines()
words[:10]
b = {}
for w in words[:1]:
    chs = ['.'] + list(w) + ['.'] #    chs = ['<S>'] + list(w) + ['<E>']#
    for ch1, ch2 in zip(chs, chs[1:]):
      bigram = (ch1, ch2)
      b[bigram] = b.get(bigram, 0) + 1
import torch
N = torch.zeros((len(stoi),len(stoi)) , dtype=torch.int32) #28#
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i, s in enumerate(chars)}
stoi['.'] = 26
#stoi['<S>'] = 26#
##stoi['<E>'] = len(stoi)##

for w in words:
    chs = ['.'] + list(w) + ['.'] ##chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
      N[stoi[ch1], stoi[ch2]] += 1
itos = {i:s for s,i in stoi.items() }

P = (N+1).float() #Smoothing icin +1 ekledik 0 log hesaplamaya kalkmasin diye Laplace smoothing dedi Copilot ilginc... #
##for i in range(len(stoi)):##
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



xs = []
ys = []
import torch.nn.functional as F
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      xs.append(stoi[ch1])
      ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
xenc = F.one_hot(xs, num_classes=len(stoi)).float()
W = torch.randn((len(stoi),len(stoi)), generator=g, requires_grad=True)
for k in range(100):
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    W.grad = None 
    loss.backward() 
    W.data += -50 * W.grad
print(loss.item())

