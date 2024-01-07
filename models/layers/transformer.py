from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.layers.encoder import EncoderLayer
from models.layers.decoder import DecoderLayer

class Transformer(nn.Module):
    """
        Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        d_model: dimensionality of model's embeddings
        num_heads: number of attention heads in multi-head attention mechanism
        num_layers: number of layers for both encoder and decoder
        d_ff: dimensionality of feedforward network
        max_seq_len: maximum sequence length, used in positional encoding
        dropout: specify probability of dropout rate
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, p_drop):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, p_drop=p_drop) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, p_drop=p_drop) for _ in range(num_layers)])

        # for processing decoder output
        self.fc = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        self.dropout = nn.Dropout(p=p_drop)

    def generate_mask(self, src, tgt):
        src_mask = 