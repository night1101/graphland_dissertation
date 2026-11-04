import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x.squeeze(-1)