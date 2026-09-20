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
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
  X_b = X[ix]
  Y_b = Y[ix]
  emb = C[Xb]
  embcat = emb.view(emb.shape[0], -1)
  hprebn = emb @ W1 + b1


############CLUSTER2############################################################



############CLUSTER3############################################################


