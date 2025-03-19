"""
Mathematical tokenization utilities.
Enhanced version of tokenization from NaturalProofs.
"""

import re
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
import transformers

def replace_math_links(text: str) -> str:
    """Replace mathematical links with their display text."""
    def _replace(line: str) -> str:
        matches = re.findall(r'(\[\[([^]]*)\]\])', line)
        for match in matches:
            full, inner = match
            splt = inner.split('|')
            if len(splt) == 1:
                txt = splt[0]
            elif len(splt) == 2:
                txt = splt[1]
            else:
                txt = ''.join(splt[1:])
            if full in line:
                line = line.replace(full, txt)
        return line
    
    lines = text.split('\n')
    lines = [_replace(line) for line in lines]
    return '\n'.join(lines)

class MathTokenizer:
    """Tokenizer for mathematical text with enhanced features."""
    
    def __init__(self, model_type: str = 'bert-base-cased'):
        """
        Initialize the math tokenizer.
        
        Args:
            model_type: The transformer model type to use for tokenization
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
        self.model_max_length = self.tokenizer.model_max_length
        self.special_tokens = {
            "sep_token": self.tokenizer.sep_token,
            "pad_token": self.tokenizer.pad_token,
            "cls_token": self.tokenizer.cls_token
        }
        
        # Log initialization
        print(f"Initialized MathTokenizer with model: {model_type}")
        print(f"Model max length: {self.model_max_length}")
    
    def tokenize_theorem(self, title: str, content: str = "", max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Tokenize a theorem with title and content.
        
        Args:
            title: The theorem title or statement
            content: Optional content or context
            max_length: Optional maximum token length
            
        Returns:
            Dictionary with tokenization results
        """
        # Clean the text
        title = replace_math_links(title)
        content = replace_math_links(content)
        
        # Combine title and content
        inputs = f"{title}{self.tokenizer.sep_token}{content}"
        
        # Tokenize
        encoding = self.tokenizer(
            inputs, 
            truncation=True,
            max_length=max_length or self.model_max_length,
            return_tensors="pt"
        )
        
        # Convert to single batch item
        input_ids = encoding['input_ids'][0] if encoding['input_ids'].dim() > 1 else encoding['input_ids']
        attention_mask = encoding['attention_mask'][0] if encoding['attention_mask'].dim() > 1 else encoding['attention_mask']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids),
            'text': inputs
        }
    
    def tokenize_proof(self, theorem_title: str, proof_text: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Tokenize a proof with theorem title and proof text.
        
        Args:
            theorem_title: The theorem title or statement
            proof_text: The proof text
            max_length: Optional maximum token length
            
        Returns:
            Dictionary with tokenization results
        """
        return self.tokenize_theorem(theorem_title, proof_text, max_length)
    
    def batch_tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts to tokenize
            max_length: Optional maximum token length
            
        Returns:
            Dictionary with batch tokenization results
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length or self.model_max_length,
            return_tensors="pt"
        )
    
    def convert_ids_to_tokens(self, ids: Union[List[int], torch.Tensor]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids: Token IDs
            
        Returns:
            List of tokens
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens to string.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed string
        """
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)