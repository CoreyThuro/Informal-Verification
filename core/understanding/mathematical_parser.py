"""
Mathematical understanding parser that uses NaturalProofs models
to extract and structure mathematical proofs.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from ir.proof_ir import ProofIR, ProofNode, NodeType, create_theorem_node, create_assumption_node, create_step_node, create_conclusion_node

# Import the NaturalProofs interface
from core.naturalproofs_integration import get_naturalproofs_interface

# Configure logging
logger = logging.getLogger(__name__)

class MathematicalParser:
    """Parser for mathematical language that converts to our IR."""
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the parser.
        
        Args:
            model_path: Optional path to pre-trained models
            use_gpu: Whether to use GPU for inference
        """
        self.np_interface = get_naturalproofs_interface(model_path, use_gpu)
        logger.info("Initialized mathematical parser with NaturalProofs")
    
    def parse_proof(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Parse a mathematical proof into structured representation.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with parsed information
        """
        # Use the NaturalProofs interface to parse the proof
        parsed = self.np_interface.parse_proof(theorem_text, proof_text)
        
        logger.debug(f"Parsed proof with domain: {parsed['domain']}, pattern: {parsed['pattern']['name']}")
        
        return parsed
    
    def convert_to_ir(self, parsed_proof: Dict[str, Any]) -> ProofIR:
        """
        Convert parsed proof to our Intermediate Representation.
        
        Args:
            parsed_proof: The parsed proof information
            
        Returns:
            ProofIR instance
        """
        # Create theorem node
        theorem_node = create_theorem_node(parsed_proof["theorem_text"])
        
        # Create proof tree from structure
        proof_tree = []
        
        # Add assumption nodes
        for assumption in parsed_proof["structure"]["assumptions"]:
            proof_tree.append(create_assumption_node(assumption))
        
        # Add step nodes
        for step in parsed_proof["structure"]["steps"]:
            proof_tree.append(create_step_node(step))
        
        # Add conclusion nodes
        for conclusion in parsed_proof["structure"]["conclusions"]:
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
        proof_ir.metadata["variables"] = parsed_proof["variables"]
        
        return proof_ir
    
    def parse_and_convert(self, theorem_text: str, proof_text: str) -> ProofIR:
        """
        Parse and convert a proof to IR in one step.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            ProofIR instance
        """
        parsed = self.parse_proof(theorem_text, proof_text)
        return self.convert_to_ir(parsed)