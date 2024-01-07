import torch
from torch import nn

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, feedforward_dim, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()

        self.linear_layer1 = nn.Linear(d_model, feedforward_dim)
        # ReLU activation
        # Rectified Linear Unit Function
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(feedforward_dim, d_model)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)

        return x
    
# sanity check - testing
if __name__ == "__main__":
    input_data = torch.randn(32, 10, 512)

    feedforward = PositionWiseFeedForwardLayer(d_model=512, feedforward_dim=2048)

    output = feedforward(input_data)

    print(output)
    # torch.Size([32, 10, 512])
    print(output.shape)