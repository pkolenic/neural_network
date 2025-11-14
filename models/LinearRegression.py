import torch
from torch import nn


# Create a linear regression model class
class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(
                1, # start with a random weight and try to adjust it to the ideal weight
                requires_grad=True, # can this parameter be updated via gradient descent?
                dtype=torch.float, # PyTorch loves the datatype torch.float32
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                1, # start with a random bias and try to adjust it to the ideal bias
                requires_grad=True,  # can this parameter be updated via gradient descent?
                dtype=torch.float,  # PyTorch loves the datatype torch.float32
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation in the model
        :param x: The input data
        :return: The linear regression formula applied to the input data
        """
        return self.weights * x + self.bias  # this is the linear regression formula
