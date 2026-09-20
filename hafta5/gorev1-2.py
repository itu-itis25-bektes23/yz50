#MLP Language Model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import math

# --- veriyi oku ---
words = open('turkce_isimler.txt', 'r').read().splitlines()
words = [w.replace('İ', 'i').replace('I', 'ı').lower().strip() for w in words]
words = [w for w in words if ' ' not in w and '.' not in w and len(w) > 0 ]
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
words_tr = words[:n1]
words_vl = words[n1:n2]
words_te = words[n2:]
# dosyadan satır satır, boşlukları temizle
print(len(words), words[:5])

# --- alfabe ---
chars = sorted(set(''.join(words)))            # tüm kelimelerdeki benzersiz harfler, sıralı
stoi = {ch: i + 1 for i, ch in enumerate(chars)}           # harf -> indeks, 1'den başlat
stoi['.'] = 0
itos = {v: k for k, v in stoi.items()}    # ters çevir
vocab_size = len(itos) # 27 diye sabit yazma
print(vocab_size)
print(sorted(stoi.keys()))
# --- veri seti ---
block_size = 3

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y
Xtr, Ytr = build_dataset(words_tr)
Xdev, Ydev = build_dataset(words_vl)
Xte, Yte = build_dataset(words_te)

print(Xtr.shape, Xtr.dtype, Ytr.shape, Ytr.dtype)
print(Xdev.shape, Xdev.dtype, Ydev.shape, Ydev.dtype)
print(Xte.shape, Xte.dtype, Yte.shape, Yte.dtype)

for x, y in zip(Xtr[:8], Ytr[:8]):
    print(''.join(itos[i.item()] for i in x), '--->', itos[y.item()])

############SETTINGS############################################################
n_emb = 100
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_emb), generator=g)
gain = 5 / 3
fan_in = block_size * n_emb
W1 = torch.randn((fan_in,200), generator=g) * (gain / math.sqrt(fan_in))
b1 = torch.randn(200) * 0.01
W2 = torch.randn((200,vocab_size), generator=g) * 0.01 
b2 = torch.zeros(vocab_size)
bngain = torch.randn((1, 200)) * 0.1 + 1
bnbias = torch.randn((1, 200)) * 0.01
bnmean_running = torch.zeros((1, 200))
bnstd_running =  torch.ones((1, 200))
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters: p.requires_grad = True
############CLUSTER1############################################################
n = 32
ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
Xb = Xtr[ix]
Yb = Ytr[ix]
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1
############CLUSTER2############################################################
bnmeani = hprebn.mean(dim=0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff **2
bnvar = bndiff2.sum(dim=0, keepdim=True) / (n - 1)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
############CLUSTER3############################################################
h = torch.tanh(hpreact)
logits = h @ W2 + b2
logit_maxes = logits.max(dim=1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = torch.exp(norm_logits)
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inv = counts_sum**(-1)
probs = counts * counts_sum_inv
logprobs = torch.log(probs)
loss = -logprobs[range(n), Yb].mean()

if (loss - F.cross_entropy(logits, Yb)).abs() < 1e-6:
  print("MATCHING")

###############KOPYA####################################
# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
###############KOPYA####################################

for p in parameters:
  p.grad = None 
tensors = [emb,embcat,h,logits,logit_maxes,norm_logits,counts,counts_sum,counts_sum_inv, logprobs, probs, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn]
for el in tensors:
  el.retain_grad()
loss.backward()
dlogprobs = torch.zeros(logprobs.grad.shape)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (counts * dprobs).sum(dim=1, keepdim=True)
dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv

cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv ', dcounts_sum_inv, counts_sum_inv )
cmp('counts_sum', dcounts_sum, counts_sum)

###########FROMNOWONDEFINITIONANDCMPONEAFTEANOTHER############
dcounts = counts_sum_inv * dprobs
dcounts += torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)
dlogit_maxes = -logit_maxes.sum(dim=1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
dlogits = dnorm_logits + F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

