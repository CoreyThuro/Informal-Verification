"""
NaturalProofs Integration Module.
Provides a simplified interface to NaturalProofs functionality.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import os
import torch

# Configure logging
logger = logging.getLogger(__name__)

class NaturalProofsInterface:
    """
    Simplified interface for NaturalProofs functionality.
    
    This class wraps the core NaturalProofs components and provides
    a simpler interface for the rest of the system to use.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the NaturalProofs interface.
        
        Args:
            model_path: Optional path to pre-trained models
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Import the core components
        try:
            from core.tokenization.tokenizer import MathTokenizer
            from core.models.pairwise_model import MathematicalModel
            
            self.tokenizer = MathTokenizer()
            self.model = MathematicalModel(model_path=model_path)
            
            # Move model to appropriate device
            if hasattr(self.model, 'x_encoder'):
                self.model.x_encoder.to(self.device)
            if hasattr(self.model, 'r_encoder'):
                self.model.r_encoder.to(self.device)
                
            self._has_models = True
            logger.info("Successfully initialized NaturalProofs models")
        except Exception as e:
            logger.warning(f"Could not initialize NaturalProofs models: {e}")
            self._has_models = False
    
    def parse_theorem(self, theorem_text: str) -> Dict[str, Any]:
        """
        Parse a theorem statement.
        
        Args:
            theorem_text: The theorem statement text
            
        Returns:
            Dictionary with parsed theorem information
        """
        if not self._has_models:
            return self._fallback_parse_theorem(theorem_text)
        
        try:
            # Tokenize the theorem
            tokens = self.tokenizer.tokenize_theorem(theorem_text)
            
            # Get the embedding
            with torch.no_grad():
                embedding = self.model.encode_theorem(tokens['input_ids'].to(self.device))
            
            # Extract variables and concepts
            variables = self._extract_variables(tokens['tokens'])
            
            # Detect domain
            domain = self._detect_domain(embedding, tokens['tokens'])
            
            return {
                "theorem_text": theorem_text,
                "embedding": embedding.cpu().numpy().tolist(),
                "tokens": tokens['tokens'],
                "variables": variables,
                "domain": domain
            }
        except Exception as e:
            logger.warning(f"Error parsing theorem with NaturalProofs: {e}")
            return self._fallback_parse_theorem(theorem_text)
    
    def parse_proof(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Parse a proof with theorem context.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with parsed proof information
        """
        if not self._has_models:
            return self._fallback_parse_proof(theorem_text, proof_text)
        
        try:
            # Tokenize the theorem and proof
            tokens = self.tokenizer.tokenize_proof(theorem_text, proof_text)
            
            # Get the embedding
            with torch.no_grad():
                embedding = self.model.encode_theorem(tokens['input_ids'].to(self.device))
            
            # Extract structure
            structure = self._extract_structure(proof_text)
            
            # Extract variables and concepts
            variables = self._extract_variables(tokens['tokens'])
            
            # Detect domain
            domain = self._detect_domain(embedding, tokens['tokens'])
            
            # Detect pattern
            pattern = self._detect_pattern(proof_text)
            
            return {
                "theorem_text": theorem_text,
                "proof_text": proof_text,
                "embedding": embedding.cpu().numpy().tolist(),
                "tokens": tokens['tokens'],
                "structure": structure,
                "variables": variables,
                "domain": domain,
                "pattern": pattern
            }
        except Exception as e:
            logger.warning(f"Error parsing proof with NaturalProofs: {e}")
            return self._fallback_parse_proof(theorem_text, proof_text)
    
    def find_similar_theorems(self, theorem_text: str, reference_theorems: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar theorems to the given theorem.
        
        Args:
            theorem_text: The theorem text
            reference_theorems: List of reference theorem texts
            top_k: Number of similar theorems to return
            
        Returns:
            List of similar theorems with similarity scores
        """
        if not self._has_models or not reference_theorems:
            return []
        
        try:
            # Use the model's retrieve_references method
            similar_refs = self.model.retrieve_references(
                theorem_text, reference_theorems, top_k=top_k
            )
            
            # Format the results
            return [
                {"text": ref, "score": score}
                for ref, score in similar_refs
            ]
        except Exception as e:
            logger.warning(f"Error finding similar theorems: {e}")
            return []
    
    def _extract_variables(self, tokens: List[str]) -> List[str]:
        """
        Extract variables from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of variables
        """
        # Simple heuristic for variables (single letters)
        variables = set()
        for token in tokens:
            if len(token) == 1 and token.isalpha():
                variables.add(token)
        
        # Add common math variables if not found
        if not variables:
            for var in ['x', 'y', 'z', 'n', 'm', 'k']:
                if var in tokens:
                    variables.add(var)
        
        return sorted(list(variables))
    
    def _extract_structure(self, proof_text: str) -> Dict[str, Any]:
        """
        Extract the structure of a proof.
        
        Args:
            proof_text: The proof text
            
        Returns:
            Dictionary with proof structure
        """
        import re
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.;\n]', proof_text) if s.strip()]
        
        # Simple structure extraction
        assumptions = []
        steps = []
        conclusions = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Identify assumptions
            if any(marker in sentence_lower for marker in ['assume', 'let', 'suppose', 'given']):
                assumptions.append(sentence)
            # Identify conclusions
            elif any(marker in sentence_lower for marker in ['therefore', 'thus', 'hence', 'so', 'conclude']):
                conclusions.append(sentence)
            # Everything else is a step
            else:
                steps.append(sentence)
        
        return {
            "assumptions": assumptions,
            "steps": steps,
            "conclusions": conclusions
        }
    
    def _detect_domain(self, embedding: torch.Tensor, tokens: List[str]) -> str:
        """
        Detect the mathematical domain based on embedding and tokens.
        
        Args:
            embedding: Theorem embedding
            tokens: Theorem tokens
            
        Returns:
            Domain code (MSC classification)
        """
        # Simple keyword-based approach as fallback
        domain_keywords = {
            "11": ["prime", "number", "divisible", "integer", "even", "odd"],
            "12-20": ["group", "ring", "field", "algebra", "vector", "matrix"],
            "26-42": ["limit", "continuous", "derivative", "integral", "function"],
            "54-55": ["topology", "open", "closed", "compact", "connected"]
        }
        
        # Count keywords
        counts = {domain: 0 for domain in domain_keywords}
        for token in tokens:
            token_lower = token.lower()
            for domain, keywords in domain_keywords.items():
                if token_lower in keywords:
                    counts[domain] += 1
        
        # Return domain with highest count, or default
        if max(counts.values()) > 0:
            return max(counts.items(), key=lambda x: x[1])[0]
        
        return "00"  # General mathematics
    
    def _detect_pattern(self, proof_text: str) -> Dict[str, Any]:
        """
        Detect the proof pattern.
        
        Args:
            proof_text: The proof text
            
        Returns:
            Dictionary with pattern information
        """
        import re
        
        proof_lower = proof_text.lower()
        
        # Check for common patterns using regex
        patterns = [
            ("induction", r'\b(induction|base\s+case|inductive|base\s+step)\b'),
            ("contradiction", r'\b(contradiction|contrary|suppose\s+not|assume\s+not)\b'),
            ("case_analysis", r'\b(case|cases|first\s+case|second\s+case)\b'),
            ("direct", r'\b(direct|straightforward)\b'),
            ("existence", r'\b(exists|construct|there\s+exists|there\s+is)\b'),
            ("uniqueness", r'\b(unique|uniqueness|exactly\s+one)\b'),
            ("evenness_proof", r'\b(even|divisible\s+by\s+2)\b')
        ]
        
        for pattern_name, regex in patterns:
            if re.search(regex, proof_lower):
                return {"name": pattern_name, "confidence": 0.9}
        
        # Default to direct proof
        return {"name": "direct", "confidence": 0.5}
    
    def _fallback_parse_theorem(self, theorem_text: str) -> Dict[str, Any]:
        """
        Fallback parsing for theorems when NaturalProofs is not available.
        
        Args:
            theorem_text: The theorem text
            
        Returns:
            Dictionary with basic parsing information
        """
        import re
        
        # Simple variable extraction
        variables = sorted(list(set(re.findall(r'\b([a-zA-Z])\b', theorem_text))))
        
        # Simple domain detection
        domain = "00"  # Default to general mathematics
        domain_keywords = {
            "11": ["prime", "number", "divisible", "integer", "even", "odd"],
            "12-20": ["group", "ring", "field", "algebra", "vector", "matrix"],
            "26-42": ["limit", "continuous", "derivative", "integral", "function"],
            "54-55": ["topology", "open", "closed", "compact", "connected"]
        }
        
        for d, keywords in domain_keywords.items():
            if any(kw in theorem_text.lower() for kw in keywords):
                domain = d
                break
        
        return {
            "theorem_text": theorem_text,
            "embedding": None,
            "tokens": theorem_text.split(),
            "variables": variables,
            "domain": domain
        }
    
    def _fallback_parse_proof(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Fallback parsing for proofs when NaturalProofs is not available.
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Dictionary with basic parsing information
        """
        # Get theorem info
        theorem_info = self._fallback_parse_theorem(theorem_text)
        
        # Extract structure
        structure = self._extract_structure(proof_text)
        
        # Detect pattern
        pattern = self._detect_pattern(proof_text)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "embedding": None,
            "tokens": theorem_info["tokens"] + proof_text.split(),
            "structure": structure,
            "variables": theorem_info["variables"],
            "domain": theorem_info["domain"],
            "pattern": pattern
        }


def get_naturalproofs_interface(model_path: Optional[str] = None, use_gpu: bool = False) -> NaturalProofsInterface:
    """
    Get a NaturalProofs interface instance.
    
    Args:
        model_path: Optional path to pre-trained models
        use_gpu: Whether to use GPU for inference
        
    Returns:
        NaturalProofs interface instance
    """
    return NaturalProofsInterface(model_path, use_gpu)