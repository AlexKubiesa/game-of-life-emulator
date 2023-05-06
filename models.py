import torch
from torch import nn
import torch.nn.functional as F

class CellPredictorModel(nn.Module):
    """Predicts the next state of the cells.

    Inputs:
        x: Tensor of shape (batch_size, channels, width+2, height+2), where channels=1. width and height are the dimensions of the entire game grid.
           We add one cell of padding on each side to ensure that predictions can be made for the boundary cells.
    
    Returns: Tensor of shape (batch_size, width, height), the logits of the predicted states.
    """

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 85, 3)
        self.conv1 = nn.Conv2d(85, 10, 1)
        self.conv2 = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        logits = self.conv2(x)
        logits = torch.squeeze(logits, 1) # Remove channels dimension
        return logits
