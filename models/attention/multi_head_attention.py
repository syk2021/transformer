from torch import nn
from models.attention.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = ScaledDotProductAttention()
        self.d_model = d_model
        self.num_heads = num_heads
        # equation in page 5
        self.d_k = d_model // num_heads

        # Linear layers for transforming inputs
        # Query Transformation
        self.W_q = nn.Linear(d_model, d_model)
        # Key Transformation
        self.W_k = nn.Linear(d_model, d_model)
        # Value Transformation
        self.W_v = nn.Linear(d_model, d_model)
        # Output Transformation
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def concat_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # linear transformation of query, key, value matrices
        # self.W_q(Q) = dot product of W_q, Q (multiplying query with query weight matrix)
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)

        # split tensor by number of heads
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)

        # scaled dot product attention to compute similarity
        # masked attention will be used in decoder so don't just set to None
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # concat heads that we split earlier
        attn_output = self.concat_heads(attn_output)
    
        # final linear transformation
        y = self.W_o(attn_output)

        return y