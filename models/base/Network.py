import torch
import torch.nn as nn
import torch.nn.functional as F


class MYNET(nn.Module):
    """
    Base network class for FCAC
    This is a minimal base class that provides the structure for the STDU MYNET to inherit from.
    """
    
    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode
        
    def forward(self, x):
        """
        Base forward pass - to be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward")
        
    def encode(self, x):
        """
        Base encoding function - to be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement encode")
