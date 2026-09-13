#MLP Language Model
import torch
import torch.nn.functional as F

# --- veriyi oku ---
words = read(notes.txt)  not = ''        # dosyadan satır satır, boşlukları temizle
print(len(words), words[:5])

# --- alfabe ---
chars = set            # tüm kelimelerdeki benzersiz harfler, sıralı
stoi = i..N             # harf -> indeks, 1'den başlat
stoi['.'] = 0
itos = xs+ itos(x)             # ters çevir
vocab_size = len(itos) # 27 diye sabit yazma

# --- veri seti ---
block_size = 3

X, Y = [], []
for w in words:
    context = [0] * blocksize
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape, X.dtype, Y.shape, Y.dtype)

for x, y in zip(X[:8], Y[:8]):
    print(''.join(sozluk[i.item()] for i in tensor), '--->', itos[y.item()])

# --- kontrol: ilk kelimenin satırlarını yazdır ---
for x, y in zip(X[:8], Y[:8]):
    print(itos(x, y))         # context harflerini itos ile çevir, '--->' , hedef harf

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, 2), generator=g)

emb = C[X]
print(emb.shape)

emb = C[X]
print(emb.shape)       # (N, 3, 2) olmalı
