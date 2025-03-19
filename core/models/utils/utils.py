# core/utils/utils.py
"""
Utility functions adapted from NaturalProofs for sequence model support.
"""

import torch
from typing import List, Any, Union

def trim(model, ids):
    """
    Trim a sequence of token IDs to remove special tokens.
    Simple implementation to match the NaturalProofs function signature.
    
    Args:
        model: The model with pad_token_id attribute
        ids: Tensor of token IDs
        
    Returns:
        Trimmed tensor of token IDs
    """
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids)
    
    # Trim padding tokens
    pad_idx = model.hparams.ypad if hasattr(model, 'hparams') else getattr(model, 'pad_idx', 0)
    mask = ids.ne(pad_idx)
    if mask.any():
        return ids[:mask.sum()]
    else:
        return ids

def extract_rankings(model, x, y, use_first=True, use_generations=True):
    """
    Extract rankings from model outputs.
    Simple implementation to match the NaturalProofs function signature.
    
    Args:
        model: The model
        x: Input tensor
        y: Output tensor
        use_first: Whether to use the first token
        use_generations: Whether to use generated tokens
        
    Returns:
        List of token IDs sorted by relevance
    """
    # For a simple implementation, just return the token IDs in y
    if use_generations:
        # If y is a tensor, convert to list
        if isinstance(y, torch.Tensor):
            return y.view(-1).tolist()
        return y
    else:
        # Return an empty list as a fallback
        return []

def get_dataloaders(tokenized_data, xpad, ypad, token_limit=16384, buffer_size=10000, 
                    workers=0, set_mode=True, order='ground-truth'):
    """
    Simplified dataloader function to avoid dependencies.
    This is a placeholder that returns empty dataloaders.
    
    Returns:
        Dictionary with empty train and valid dataloaders
    """
    # In a real implementation, this would create PyTorch dataloaders
    # For now, we return a structure that matches what's expected
    return {
        'train': [],
        'valid': []
    }

def get_idx2tok_map(pretrained_model_idx2rid, dataset_rid2tok):
    """
    Get a mapping from model indices to token indices.
    Simplified version that returns an empty mapping.
    
    Returns:
        Empty dictionary
    """
    # In a real implementation, this would create a mapping
    return {}