"""
Pairwise model for theorem-reference retrieval.
Enhanced version of the NaturalProofs model adapted for our codebase.
"""

import torch
import torch.nn.functional as F
import transformers
from typing import List, Dict, Any, Tuple, Optional, Union

from core.models.base_model import BaseModel

class MathematicalModel(BaseModel):
    """
    Mathematical understanding model for theorem and reference encoding.
    Based on NaturalProofs pairwise model architecture but simplified for our needs.
    """
    
    def __init__(self, model_type: str = 'bert-base-cased', checkpoint_path: Optional[str] = None):
        """
        Initialize the mathematical model.
        
        Args:
            model_type: The transformer model type to use
            checkpoint_path: Optional path to a pre-trained model checkpoint
        """
        super().__init__(model_type, checkpoint_path)
    
    def _initialize_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Initialize the model components.
        
        Args:
            checkpoint_path: Optional path to a checkpoint
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_type)
        self.pad_idx = self.tokenizer.pad_token_id
        
        # Initialize encoders
        self.x_encoder = transformers.AutoModel.from_pretrained(self.model_type)
        self.r_encoder = transformers.AutoModel.from_pretrained(self.model_type)
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint and "state_dict" in checkpoint:
                self._load_weights(checkpoint["state_dict"])
            
        # Set model to evaluation mode
        self.x_encoder.eval()
        self.r_encoder.eval()
    
    def _load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load weights from a state dictionary.
        
        Args:
            state_dict: The state dictionary containing model weights
        """
        # Process state dict to match our model structure
        x_encoder_dict = {}
        r_encoder_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('x_encoder.'):
                x_encoder_dict[k.replace('x_encoder.', '')] = v
            elif k.startswith('r_encoder.'):
                r_encoder_dict[k.replace('r_encoder.', '')] = v
        
        # Load weights
        if x_encoder_dict:
            self.x_encoder.load_state_dict(x_encoder_dict)
        if r_encoder_dict:
            self.r_encoder.load_state_dict(r_encoder_dict)
    
    def encode_theorem(self, x: Union[str, torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Encode a theorem into a vector representation.
        
        Args:
            x: The theorem text, token IDs, or tensor
            
        Returns:
            Theorem embedding tensor
        """
        # Convert to tensor if needed
        if isinstance(x, str):
            x = self.tokenizer(x, return_tensors='pt')['input_ids']
        elif isinstance(x, list):
            x = torch.tensor([x])
        
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected string, list or tensor, got {type(x)}")
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Create attention mask
        xmask = x.ne(self.pad_idx).float()
        
        # Get encoding
        with torch.no_grad():
            outputs = self.x_encoder(x, attention_mask=xmask)
            x_enc = outputs[0][:, 0]  # Use [CLS] token embedding
        
        return x_enc
    
    def encode_reference(self, r: Union[str, torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Encode a reference into a vector representation.
        
        Args:
            r: The reference text, token IDs, or tensor
            
        Returns:
            Reference embedding tensor
        """
        # Convert to tensor if needed
        if isinstance(r, str):
            r = self.tokenizer(r, return_tensors='pt')['input_ids']
        elif isinstance(r, list):
            r = torch.tensor([r])
        
        # Ensure r is a tensor
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"Expected string, list or tensor, got {type(r)}")
        
        # Add batch dimension if needed
        if r.dim() == 1:
            r = r.unsqueeze(0)
        
        # Create attention mask
        rmask = r.ne(self.pad_idx).float()
        
        # Get encoding
        with torch.no_grad():
            outputs = self.r_encoder(r, attention_mask=rmask)
            r_enc = outputs[0][:, 0]  # Use [CLS] token embedding
        
        return r_enc
    
    def encode(self, text: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text into a vector representation.
        Uses the theorem encoder by default.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded representation
        """
        if isinstance(text, list) and all(isinstance(t, str) for t in text):
            # Batch of strings
            encodings = []
            for t in text:
                encodings.append(self.encode_theorem(t))
            return torch.stack(encodings)
        else:
            # Single item
            return self.encode_theorem(text)
    
    def compute_similarity(self, x_enc: torch.Tensor, r_enc: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between theorem and reference encodings.
        
        Args:
            x_enc: Theorem encoding
            r_enc: Reference encoding
            
        Returns:
            Similarity scores
        """
        # Compute dot product similarity
        similarity = x_enc.matmul(r_enc.transpose(0, 1))
        return similarity
    
    def retrieve_references(self, theorem: str, references: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant references for a theorem.
        
        Args:
            theorem: The theorem text
            references: List of reference texts
            top_k: Number of top matches to return
            
        Returns:
            List of (reference, score) tuples
        """
        # Encode theorem
        x_enc = self.encode_theorem(theorem)
        
        # Encode all references
        r_encs = []
        for ref in references:
            r_enc = self.encode_reference(ref)
            r_encs.append(r_enc)
        
        if not r_encs:
            return []
        
        r_encs = torch.cat(r_encs, dim=0)
        
        # Compute similarities
        similarities = self.compute_similarity(x_enc, r_encs)
        
        # Get top-k references
        k = min(top_k, len(references))
        if k == 0:
            return []
            
        top_values, top_indices = similarities[0].topk(k)
        
        return [(references[i], top_values[j].item()) for j, i in enumerate(top_indices)]
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the model to a device.
        
        Args:
            device: The device to move to
        """
        self.x_encoder.to(device)
        self.r_encoder.to(device)