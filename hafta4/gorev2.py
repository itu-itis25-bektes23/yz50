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

emb = C[X]
emb = (N, 6)
print(emb.shape)

W1 = g
b1 = g
W2 = g
b2 = g
W_h = 100
b_h = 100
torch.tanh(W_h*x + b_h)
F.cross_entropy(W_h, b_h) ?= -torch.log(probabilities)
