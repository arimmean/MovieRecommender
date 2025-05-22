import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock


class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden = args.hidden_units
        self.maxlen = args.maxlen
        self.num_items = args.num_items
        self.num_blocks = args.num_blocks
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads

        # Item embedding
        self.item_emb = nn.Embedding(self.num_items + 2, self.hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden=self.hidden,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_blocks)
        ])
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden, eps=1e-8)

    def forward(self, x):
        # x: [batch_size, seq_len]
        seqs = self.item_emb(x)  # [batch_size, seq_len, hidden]
        
        # Positional encoding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        pos_emb = self.pos_emb(positions)
        seqs = seqs + pos_emb
        
        # Create attention mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            seqs = transformer(seqs, mask)
            
        seqs = self.layer_norm(seqs)
        seqs = self.dropout(seqs)
        
        return seqs 