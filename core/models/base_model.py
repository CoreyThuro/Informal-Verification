"""
Base model class for mathematical understanding models.
Provides common functionality for all models in the core module.
"""

import torch
from typing import Dict, List, Any, Optional, Union

class BaseModel:
    """
    Base class for all mathematical understanding models.
    """
    
    def __init__(self, model_type: str = 'bert-base-cased', checkpoint_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_type: The transformer model type to use
            checkpoint_path: Optional path to a checkpoint
        """
        self.model_type = model_type
        self._initialize_model(checkpoint_path)
    
    def _initialize_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Initialize the model.
        
        Args:
            checkpoint_path: Optional path to a checkpoint
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with default weights")
            return None
    
    def encode(self, text: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text into a vector representation.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded representation
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the model to a device.
        
        Args:
            device: The device to move to
        """
        raise NotImplementedError("Subclasses must implement this method")