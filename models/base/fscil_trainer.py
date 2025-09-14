"""
Base FSCIL Trainer
This is a minimal base trainer that the STDU trainer can inherit from.
The actual implementation is in models/stdu/base.py
"""

# Since the actual Trainer is in stdu/base.py, we import it from there
from models.stdu.base import Trainer as FSCILTrainer
