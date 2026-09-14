#MLP Language Model
import torch
import torch.nn.functional as F

# --- veriyi oku ---
words = open('isimler.txt').read().splitlines()      # dosyadan satır satır, boşlukları temizle
print(len(words), words[:5])

# --- alfabe ---
chars = sorted(set(''.join(words)))            # tüm kelimelerdeki benzersiz harfler, sıralı
stoi = {ch: i + 1 for i, ch in enumerate(chars)}           # harf -> indeks, 1'den başlat
stoi['.'] = 0
itos = {v: k for k, v in stoi.items()}    # ters çevir
vocab_size = len(itos) # 27 diye sabit yazma

# --- veri seti ---
block_size = 3

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
print(X.shape, X.dtype, Y.shape, Y.dtype)

for x, y in zip(X[:8], Y[:8]):
    print(''.join(itos[i.item()] for i in x), '--->', itos[y.item()])


g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, 2), generator=g)


N = X.shape[0]
emb = C[X]
emb = emb.reshape(N, 6)
print(emb.shape)

W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

h = torch.tanh(emb @ W1 + b1)
logits = h @ W2 + b2
counts = torch.exp(logits)
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(N), Y].log().mean()
print(torch.allclose(F.cross_entropy(logits, Y), loss))

X_slice = X[1..32]
Y_slice = Y[1..32]
emb_slice = C[X_slice]
emb_slice = emb.reshape(N, 6)
print(emb.shape)

h = torch.tanh(emb @ W1 + b1)
logits = h @ W2 + b2

loss = F.cross_entropy(logits, Y_slice)
for elements i in parameters 
  set p.grad = None
loss.backward()
for elements in parameters
  p.data += -lr * p.grad
