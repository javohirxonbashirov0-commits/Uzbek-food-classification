# Uzbek-food-classification
Transformer Arxitekturasi: Nazariyadan Kodgacha
Ushbu qo'llanma matnli (yoki tasvirli) ma'lumotni Transformer modelining "yuragi" bo'lgan Attention mexanizmigacha bo'lgan jarayonini tushuntiradi.

1. Jarayonlar Ketma-ketligi
Input → Tokenization → Embedding → Positional Encoding → Multi-Head Attention → Output

2. Tokenization (Bo'laklash)
Nima bu? Matnni (yoki rasmni) kichik bo'laklarga bo'lish.

Matnda: So'zlar yoki harf birikmalari.

Rasmlarda (ViT): Rasmni kichik kvadratlarga (patch) bo'lish.

3. Word Embedding (Vektorga aylantirish)
Nima bu? So'zlarga matematik "ma'no" berish. So'zlar ko'p o'lchovli fazoda joylashadi.

Mantiq: "Muzqaymoq" va "Muzdek" vektorlari bir-biriga yaqin turadi.

4. Positional Encoding (Tartib berish)
Nima bu? Transformer hamma narsani parallel o'qigani uchun tartibni bilmaydi. Sinus va Kosinus funksiyalari orqali har bir tokenning o'rnini belgilaymiz.

Python
import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # To'lqin chastotasini hisoblash
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Juft o'rinlar
        pe[:, 1::2] = torch.cos(position * div_term) # Toq o'rinlar
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
5. Multi-Head Attention (Diqqat mexanizmi)
Nima bu? Modelning gapdagi (yoki rasmdagi) muhim qismlarga e'tibor qaratishi.

Query (Q): Men nima qidiryapman?

Key (K): Kimda javob bor?

Value (V): Topilgan ma'lumotning o'zi.

Python
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Wq = torch.nn.Linear(d_model, d_model)
        self.Wk = torch.nn.Linear(d_model, d_model)
        self.Wv = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        # Kallalarga bo'lish (view va transpose)
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        
        # Kallalarni birlashtirish
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, C)
        return self.fc(out)
6. Xulosa
Transformer modeli — bu xuddi odamdek gapni (yoki rasmni) bir butun holatda ko'ra oladigan, lekin uning har bir bo'lagiga alohida e'tibor (Attention) bera oladigan juda kuchli arxitekturadir.

Tayyorladi: Javohirxon Bashirov
