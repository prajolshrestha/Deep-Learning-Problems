import torch as t
from torch import nn
class LinearRegressionModelV2(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x: t.tensor) -> t.tensor:
        return self.linear_layer(x)
