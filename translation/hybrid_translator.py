"""
Hybrid translator that combines rule-based and LLM approaches.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import os
import re

from ir.proof_ir import ProofIR
from backends.backend_interface import BackendRegistry

logger = logging.getLogger("proof_translator")

class HybridTranslator:
    """
    Hybrid translator that uses LLM or rule-based approaches as appropriate.
    """
    
    def __init__(self, use_llm: bool = False, target_prover: str = "coq"):
        """
        Initialize the hybrid translator.
        
        Args:
            use_llm: Whether to use LLM assistance
            target_prover: The target theorem prover
        """
        self.use_llm = use_llm
        self.target_prover = target_prover
        
        # Get the backend
        try:
            self.backend = BackendRegistry.get_backend(target_prover)
        except ValueError as e:
            logger.error(f"Backend error: {str(e)}")
            raise
        
        # Check LLM configuration if needed
        self.llm_configured = False
        if use_llm:
            try:
                from llm.openai_client import verify_openai_setup
                self.llm_configured, message = verify_openai_setup()
                logger.info(f"LLM configuration: {message}")
            except ImportError:
                logger.warning("OpenAI client not available.")
    
    def translate(self, proof_ir: ProofIR) -> str:
        """
        Translate the proof IR to the target language.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The translated proof
        """
        # If LLM is enabled and configured, try direct translation
        if self.use_llm and self.llm_configured:
            logger.info("Using direct LLM translation")
            
            # Extract theorem and proof from IR
            theorem = proof_ir.original_theorem
            proof = proof_ir.original_proof
            
            try:
                # Try to use OpenAI directly
                from llm.openai_client import translate_proof_with_openai
                return translate_proof_with_openai(theorem, proof, self.target_prover)
            except ImportError:
                # Fall back to generic LLM interface
                from llm.llm_interface import translate_proof_with_llm
                return translate_proof_with_llm(theorem, proof, self.target_prover)
            except Exception as e:
                logger.warning(f"LLM translation failed: {str(e)}. Falling back to rule-based.")
        
        # Otherwise, use the rule-based backend translation
        logger.info("Using rule-based translation")
        return self.translate_with_rules(proof_ir)
    
    def translate_with_rules(self, proof_ir: ProofIR) -> str:
        """
        Translate using only rule-based approaches.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The translated proof
        """
        logger.info(f"Using rule-based translation with {self.target_prover} backend")
        return self.backend.translate(proof_ir)


# Standalone functions for use in other modules

def translate_proof(proof_ir: ProofIR, target_prover: str, use_llm: bool = False) -> str:
    """
    Translate a proof using the hybrid translator.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        use_llm: Whether to use LLM assistance
        
    Returns:
        The translated proof
    """
    translator = HybridTranslator(use_llm=use_llm, target_prover=target_prover)
    return translator.translate(proof_ir)