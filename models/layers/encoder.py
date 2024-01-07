from torch import nn
from models.attention.multi_head_attention import MultiHeadAttention
from models.attention.feed_forward import PositionWiseFeedForwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_drop):
        super(EncoderLayer, self).__init__()

        # 1 self attention layer
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)

        # 1 feedforward layer
        self.feed_forward = PositionWiseFeedForwardLayer(d_model=d_model, feedforward_dim=d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p_drop)

    def forward(self, x, mask):
        orig_1x = x
        # compute self attention (uses MultiHeadAttention class)
        x = self.self_attn(Q=x, K=x, V=x, mask=mask)

        # add & norm
        x = self.dropout1(x)
        # residual connection
        x = self.norm1(x + orig_1x)

        # 1 FF Layer
        # save x at this point in orig_2x
        orig_2x = x
        x = self.feed_forward(x)

        # add & norm
        x = self.dropout2(x)
        # residual connection
        x = self.norm2(x + orig_2x)

        return x
