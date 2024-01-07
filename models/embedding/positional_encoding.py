import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        # tensor with zeroes, to be populated with positional encodings
        self.encoding = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # page 6 equation
        # step=2 (2*i in equation)
        term_2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (term_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (term_2i / d_model)))
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        # adds positional encoding to the end of x
        x = x + self.encoding[:, :seq_len, :]
        return x

if __name__ == "__main__":
    # input tensor with shape [batch_size, seq_len, d_model (embed_size)]
    input_data = torch.randn(32, 10, 512)

    positional_encoder = PositionalEncoding(d_model=512, max_seq_len=100)
    output = positional_encoder(input_data)
    print(output)
    # torch.Size([32, 10, 512])
    print(output.shape)