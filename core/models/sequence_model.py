# core/models/sequence_model.py
"""
Sequence retrieval model from NaturalProofs.
Simplified version without external dependencies.
"""

import torch
import torch.nn.functional as F
import transformers
from typing import List, Dict, Any, Optional, Union

# Import our local utils implementation instead of NaturalProofs
import core.models.utils.utils as utils

class SequenceModel:
    """Simplified version of the SequenceRetriever for inference only."""
    
    def __init__(self, model_type='bert-base-cased', checkpoint_path=None):
        """
        Initialize the sequence model.
        
        Args:
            model_type: The transformer model type to use
            checkpoint_path: Optional path to a checkpoint
        """
        self.model_type = model_type
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
        self.pad_idx = self.tokenizer.pad_token_id
        
        # Create encoder-decoder model
        encoder_decoder_config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig.from_pretrained(model_type),
            transformers.BertConfig.from_pretrained(model_type)
        )
        
        # Set pad token IDs
        encoder_decoder_config.encoder.pad_token_id = self.pad_idx
        encoder_decoder_config.decoder.pad_token_id = self.pad_idx
        
        # Initialize model
        self.encdec = transformers.EncoderDecoderModel(encoder_decoder_config)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.encdec.load_state_dict(checkpoint['state_dict'])
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with default weights")
    
    def generate(self, x, max_length=20):
        """
        Generate a sequence from input tokens.
        
        Args:
            x: Input token IDs
            max_length: Maximum length to generate
            
        Returns:
            Generated token IDs
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x])
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Create attention mask
        xmask = x.ne(self.pad_idx).float()
        
        # Generate output
        with torch.no_grad():
            output = self.encdec.generate(
                input_ids=x,
                attention_mask=xmask,
                max_length=max_length,
                no_repeat_ngram_size=1,
                do_sample=False,
                num_beams=1
            )
        
        return output