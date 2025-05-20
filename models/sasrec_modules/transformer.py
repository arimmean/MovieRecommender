import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, num_heads, dropout_rate):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        
        self.q_linear = nn.Linear(hidden, hidden)
        self.k_linear = nn.Linear(hidden, hidden)
        self.v_linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden, hidden)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden)
        
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, hidden, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden * 4)
        self.linear2 = nn.Linear(hidden * 4, hidden)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden, num_heads, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(hidden, num_heads, dropout_rate)
        self.feed_forward = FeedForward(hidden, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x 