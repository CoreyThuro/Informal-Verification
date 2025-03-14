"""
Mathematical understanding parser that uses NaturalProofs models
to extract and structure mathematical proofs.
"""

import torch
import re
from typing import List, Dict, Any

from core.models.pairwise_model import MathematicalModel
from core.tokenization.tokenizer import MathTokenizer

class MathematicalParser:
    """Parser for mathematical language that converts to our IR."""
    
    def __init__(self, model_path=None, model_type='bert-base-cased'):
        self.model = MathematicalModel(model_type, model_path)
        self.tokenizer = MathTokenizer(model_type)
    
    def parse_proof(self, theorem_text, proof_text):
        """Parse a mathematical proof into structured representation."""
        # Tokenize the input
        tokenized = self.tokenizer.tokenize_proof(theorem_text, proof_text)
        
        # Get the embeddings
        with torch.no_grad():
            theorem_embedding = self.model.encode_theorem(tokenized['input_ids'])
        
        # Extract structure and concepts
        structure = self._extract_structure(proof_text)
        concepts = self._extract_concepts(tokenized['tokens'], theorem_embedding)
        domain = self._detect_domain(theorem_embedding, tokenized['tokens'])
        pattern = self._detect_pattern(proof_text)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "embedding": theorem_embedding,
            "structure": structure,
            "concepts": concepts,
            "domain": domain,
            "pattern": pattern
        }
    
    def _extract_structure(self, proof_text):
        """Extract the logical structure of the proof."""
        sentences = [s.strip() for s in re.split(r'[.;]', proof_text) if s.strip()]
        
        # Simple structure extraction
        steps = []
        assumptions = []
        conclusions = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in ['assume', 'let', 'suppose', 'given']):
                assumptions.append((i, sentence))
            elif any(marker in sentence_lower for marker in ['therefore', 'thus', 'hence', 'so']):
                conclusions.append((i, sentence))
            else:
                steps.append((i, sentence))
        
        return {
            "assumptions": assumptions,
            "steps": steps,
            "conclusions": conclusions
        }
    
    def _extract_concepts(self, tokens, embedding):
        """Extract mathematical concepts from the tokens."""
        # Extract variables (single letters)
        variables = []
        for token in tokens:
            if len(token) == 1 and token.isalpha():
                variables.append(token)
        
        # Extract mathematical operations
        operations = []
        op_tokens = ["+", "-", "*", "/", "=", "<", ">"]
        for token in tokens:
            if token in op_tokens:
                operations.append(token)
        
        return {
            "variables": list(set(variables)),
            "operations": list(set(operations))
        }
    
    def _detect_domain(self, embedding, tokens):
        """Detect the mathematical domain of the theorem."""
        # This would use NaturalProofs embeddings to classify the domain
        # For now, a simple rule-based approach
        domain_keywords = {
            "11": ["prime", "number", "divisible", "integer", "even", "odd"],
            "12-20": ["group", "ring", "field", "algebra", "vector"],
            "26-42": ["limit", "continuous", "derivative", "integral"],
            "54-55": ["topology", "open", "closed", "compact"]
        }
        
        counts = {domain: 0 for domain in domain_keywords}
        for token in tokens:
            for domain, keywords in domain_keywords.items():
                if token.lower() in keywords:
                    counts[domain] += 1
        
        # Return domain with highest keyword count
        if max(counts.values()) > 0:
            return max(counts.items(), key=lambda x: x[1])[0]
        
        return "00"  # General mathematics
    
    def _detect_pattern(self, proof_text):
        """Detect the proof pattern used."""
        proof_lower = proof_text.lower()
        
        # Check for common patterns
        if re.search(r'\b(induction|base\s+case|inductive)\b', proof_lower):
            return {"name": "induction", "confidence": 0.9}
        elif re.search(r'\b(contradiction|contrary|assume.*not)\b', proof_lower):
            return {"name": "contradiction", "confidence": 0.9}
        elif re.search(r'\b(case|cases|first\s+case|second\s+case)\b', proof_lower):
            return {"name": "case_analysis", "confidence": 0.9}
        elif re.search(r'\b(even|divisible\s+by\s+2)\b', proof_lower):
            return {"name": "evenness_proof", "confidence": 0.9}
        
        return {"name": "direct", "confidence": 0.7}
    
    def convert_to_ir(self, parsed_proof):
        """Convert parsed proof to our Intermediate Representation."""
        from ir.proof_ir import ProofIR, ProofNode, NodeType, create_theorem_node, create_assumption_node, create_step_node, create_conclusion_node
        
        # Create theorem node
        theorem_node = create_theorem_node(parsed_proof["theorem_text"])
        
        # Create proof tree from structure
        proof_tree = []
        
        # Add assumption nodes
        for _, assumption in parsed_proof["structure"]["assumptions"]:
            proof_tree.append(create_assumption_node(assumption))
        
        # Add step nodes
        for _, step in parsed_proof["structure"]["steps"]:
            proof_tree.append(create_step_node(step))
        
        # Add conclusion nodes
        for _, conclusion in parsed_proof["structure"]["conclusions"]:
            proof_tree.append(create_conclusion_node(conclusion))
        
        # Create ProofIR
        proof_ir = ProofIR(
            theorem=theorem_node,
            proof_tree=proof_tree,
            domain={"primary_domain": parsed_proof["domain"]},
            pattern=parsed_proof["pattern"],
            original_theorem=parsed_proof["theorem_text"],
            original_proof=parsed_proof["proof_text"],
            tactics=[]  # Will be filled later
        )
        
        # Add metadata
        proof_ir.metadata["variables"] = parsed_proof["concepts"]["variables"]
        
        return proof_ir