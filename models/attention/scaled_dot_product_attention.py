import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    """
        Reference the Scaled Dot Product Attention diagram in page 4.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, query, key, value, mask=None):
        # calculate scaled attn_scores; matmul operation replaced with einsum
        attn_scores = torch.einsum('bqd, bkd -> bqk', query, key) / torch.sqrt(query.size(-1))

        # optional mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # softmax to obtain attention weights
        attn_weights = nn.softmax(attn_scores, dim=-1)

        # matmul operation replaced with einsum
        output = torch.einsum('bqk, bkd->bqd', attn_weights, value)

        return output