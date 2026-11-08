import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, block_size, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.positions_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList()

        for block in range(num_layers):
            self.blocks.append(TransformerBlock(embedding_dim, num_heads))

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device = x.device)
        positions_vectors = self.positions_embedding(positions)
        embedding = self.token_embedding(x)
        emb_pos = positions_vectors + embedding
        out = emb_pos

        for block in self.blocks:
            out = block(out)
        logits = self.lm_head(out)
        return logits
class Attention(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias = False)
        self.key = nn.Linear(embedding_dim, head_size, bias = False)
        self.value = nn.Linear(embedding_dim, head_size, bias = False)
        self.head_size =  head_size

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / (self.head_size) ** 0.5
        B, T, C = x.shape
        mask = torch.ones(T, T, device = x.device)
        mask = torch.tril(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim = -1)
        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList()
        head_size = embedding_dim // num_heads

        for i in range(num_heads):
            self.heads.append(Attention(embedding_dim, head_size))

    def forward (self, x):
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        out = torch.cat(outputs, dim = -1)
        return out

class FeedForward(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear( embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4,  embedding_dim)
        )

    def forward(self, x):
        vectors = self.layers (x)
        return vectors

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer1 = nn.LayerNorm(embedding_dim)
        self.layer2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x):
        normalized = self.layer1(x)
        attention_out = self.attention(normalized)
        out = attention_out + x
        normalized2 = self.layer2(out)
        ffn_out = self.ff(normalized2)
        out = ffn_out + out
        return out
