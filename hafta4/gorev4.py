#MLP Language Model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# --- veriyi oku ---
words = open('names.txt').read().splitlines()  
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

n_emb = 2

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_emb), generator=g)

W1 = torch.randn((block_size * n_emb,200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200,vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
lossi = []
for i in range(30000):
  lr = 0.1
  if i > 20000:
    lr = 0.01
  ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
  X_slice = Xtr[ix]
  Y_slice = Ytr[ix]
  emb_slice = C[X_slice]
  emb_slice = emb_slice.view(emb_slice.shape[0], -1)
  h = torch.tanh(emb_slice @ W1 + b1)
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

plt.plot(lossi)
plt.show()
with torch.no_grad():
    emb_tr =  C[Xtr]          # C indexed by Xtr
    flat_tr = emb_tr.view(emb_tr.shape[0], -1)         # .view(...)
    h_tr = torch.tanh(flat_tr @ W1 + b1)            # tanh
    logits_tr = h_tr @ W2 + b2
    print('train', F.cross_entropy(logits_tr, Ytr).item())

    emb_dev =  C[Xdev]          # C indexed by Xdev
    flat_dev = emb_dev.view(emb_dev.shape[0], -1)         # .view(...)
    h_dev = torch.tanh(flat_dev @ W1 + b1)            # tanh
    logits_dev = h_dev @ W2 + b2
    print('dev', F.cross_entropy(logits_dev, Ydev).item())

for i in range(vocab_size):
  plt.annotate((C.data[i], C.data[i]),itos[i], ha='center', va='center')
  plt.scatter(C.data[0], C.data[1])
  plt.grid()
