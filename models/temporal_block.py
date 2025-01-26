import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        """
        Initializes a temporal block for a Temporal Convolutional Network (TCN).

        Args:
            in_channels (int): Number of input channels (features) to the block.
            out_channels (int): Number of output channels (features) from the block.
            kernel_size (int): Size of the convolutional kernel (filter).
            dilation (int): Dilation rate for the convolutional layers.
            dropout (float, optional): Dropout probability for regularization. Defaults to 0.2.
        """
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 =  weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 =  weight_norm(nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding, 
            dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the temporal block.
        """
        residual = x
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)