import torch
from torch import nn
from models.attention.multi_head_attention import MultiHeadAttention
from models.attention.feed_forward import PositionWiseFeedForwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_drop):
        super(DecoderLayer, self).__init__()

        # 1 self attention layer
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)

        # 1 cross attention layer (encoder - decoder)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

        # 1 feedforward layer
        self.feed_forward = PositionWiseFeedForwardLayer(d_model=d_model, feedforward_dim=d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p_drop)

    def forward(self, x, enc_output, source_mask, target_mask):
        # MASKED MULTIHEAD ATTENTION
        orig_1x = x
        # self attention
        x = self.self_attn(Q=x, K=x, V=x, mask=target_mask)
        # add & norm
        x = self.dropout1(x)
        ## residual connection
        x = self.norm1(x + orig_1x)

        # CROSS ATTENTION
        # save x at this point in orig_2x
        # cross attention
        orig_2x = x
        x = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=source_mask)
        # add & norm
        x = self.dropout2(x)
        ## residual connection
        x = self.norm2(x + orig_2x)

        # 1 FF Layer
        # save x at this point in orig_3x
        orig_3x = x
        x = self.feed_forward(x)
        # add & norm
        x = self.dropout3(x)
        ## residual connection
        x = self.norm3(x + orig_3x)

        return x
