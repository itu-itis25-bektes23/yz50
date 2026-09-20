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
n_emb = 100000
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_emb), generator=g)
gain = 5 / 3
fan_in = block_size * n_emb
W1 = torch.randn((fan_in,200), generator=g) * (gain / math.sqrt(fan_in))
b1 = torch.randn(200, generator=g) * 0.01
W2 = torch.randn((200,vocab_size), generator=g) * 0.01
b2 = torch.zeros(vocab_size)
bngain = torch.ones((1, 200))
bnbias = torch.zeros((1, 200))
bnmean_running = torch.zeros((1, 200))
bnstd_running =  torch.ones((1, 200))
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True
lossi = []



######################### TRAINING #############################################
for i in range(30000):
  lr = 0.1
  if i > 20000:
    lr = 0.01
  ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
  X_slice = Xtr[ix]
  Y_slice = Ytr[ix]
  emb_slice = C[X_slice]
  emb_slice = emb_slice.view(emb_slice.shape[0], -1)
  hpreact = emb_slice @ W1 + b1
  bnmeani = hpreact.mean(dim=0, keepdim=True)
  bnstdi = hpreact.std(dim=0, keepdim=True)
  hpreact = bngain * ((hpreact - bnmeani) / bnstdi) + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Y_slice)
  lossi.append(loss.item())
  for p in parameters:
    p.grad = None
  loss.backward()
  for p in parameters:
    p.data += -lr * p.grad
  if i % 100 == 0:
    print(i, loss.item())
  with torch.no_grad(): #########UPDATERUNNINGAVERAGEANDSTANDARDDERIVATION######
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi

#####################EVALUATING#################################################

plt.plot(lossi)
plt.show()
with torch.no_grad():
    emb_tr =  C[Xtr]          # C indexed by Xtr
    flat_tr = emb_tr.view(emb_tr.shape[0], -1)         # .view(...)
    hpreact_tr = flat_tr @ W1 + b1
    hpreact_tr = bngain * ((hpreact_tr - bnmean_running) / bnstd_running) + bnbias
    h_tr = torch.tanh(hpreact_tr)            # tanh
    logits_tr = h_tr @ W2 + b2
    print('train', F.cross_entropy(logits_tr, Ytr).item())

    emb_dev =  C[Xdev]          # C indexed by Xdev
    flat_dev = emb_dev.view(emb_dev.shape[0], -1)         # .view(...)
    hpreact_dev = flat_dev @ W1 + b1
    hpreact_dev = bngain * ((hpreact_dev - bnmean_running) / bnstd_running) + bnbias
    h_dev = torch.tanh(hpreact_dev)            # tanh
    logits_dev = h_dev @ W2 + b2
    print('dev', F.cross_entropy(logits_dev, Ydev).item())
plt.figure()
for i in range(vocab_size):
  plt.annotate(itos[i], (C.data[i,0], C.data[i,1]), ha='center', va='center')
plt.scatter(C.data[:, 0], C.data[:, 1])
plt.grid()
plt.show()


##############################SAMPLING##########################################
with torch.no_grad():
  names = 5
  for name in range(names):
    out = []
    context = [0] * block_size
    while True:
      emb_c =  C[context]          # C indexed by Xtr
      flat_c = emb_c.view(1, -1)
      hpreact_c = flat_c @ W1 + b1
      hpreact_c = bngain * ((hpreact_c - bnmean_running) / bnstd_running) + bnbias         # .view(...)
      h_c = torch.tanh(hpreact_c)            # tanh
      logits_c = h_c @ W2 + b2
      probs_c = F.softmax(logits_c, dim=1)
      ix = torch.multinomial(probs_c, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0: break
    print(''.join(itos[i] for i in out))


plt.hist(h.view(-1).tolist(), 50)
plt.show()
plt.imshow(h.abs()> 0.99, cmap='gray')
plt.show()

