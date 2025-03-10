"""
Main translation module for the proof translation system.
"""

import logging
from typing import Dict, Any, Optional
import os

from ir.proof_ir import ProofIR
from backends.backend_interface import BackendRegistry

# Configure logging
logger = logging.getLogger("proof_translator")

class ProofTranslator:
    """Main translation system for informal to formal proofs."""
    
    def __init__(self, target_prover: str = "coq", use_llm: bool = False):
        """
        Initialize the proof translator.
        
        Args:
            target_prover: The target theorem prover
            use_llm: Whether to use LLM assistance
        """
        self.target_prover = target_prover
        self.use_llm = use_llm
        
        # Check if LLM is available and configured if needed
        self.llm_configured = False
        if use_llm:
            try:
                from llm.openai_client import verify_openai_setup
                self.llm_configured, message = verify_openai_setup()
                logger.info(f"LLM configuration: {message}")
            except ImportError:
                logger.warning("LLM module not available or OpenAI API key not set")
        
        # Get the backend
        try:
            self.backend = BackendRegistry.get_backend(target_prover)
            logger.info(f"Using {target_prover} backend: {self.backend.name}")
        except ValueError as e:
            logger.error(f"Backend error: {str(e)}")
            raise
    
    def translate(self, theorem: str, proof: str) -> Dict[str, Any]:
        """
        Translate an informal proof to a formal proof.
        
        Args:
            theorem: The theorem statement
            proof: The proof text
            
        Returns:
            Dictionary with translation results
        """
        logger.info(f"Translating proof: target_prover={self.target_prover}, use_llm={self.use_llm}")
        
        # Try LLM translation first if enabled
        if self.use_llm and self.llm_configured:
            formal_proof = self._translate_with_llm(theorem, proof)
        else:
            # Parse the proof and use rule-based translation
            formal_proof = self._translate_with_rules(theorem, proof)
        
        # Verify the proof
        verification_success, error_message = False, None
        if self.backend.is_installed():
            verification_success, error_message = self.backend.verify(formal_proof)
            
            # If LLM translation failed verification, try rule-based as fallback
            if not verification_success and self.use_llm and self.llm_configured:
                logger.info("LLM translation failed verification. Trying rule-based fallback.")
                rule_based_proof = self._translate_with_rules(theorem, proof)
                
                # Verify the rule-based translation
                rb_success, rb_error = self.backend.verify(rule_based_proof)
                
                # If rule-based succeeds where LLM failed, use it instead
                if rb_success:
                    formal_proof = rule_based_proof
                    verification_success = rb_success
                    error_message = None
                    logger.info("Rule-based fallback succeeded where LLM failed.")
        
        return {
            "formal_proof": formal_proof,
            "verification_success": verification_success,
            "error_message": error_message
        }
    
    def _translate_with_llm(self, theorem: str, proof: str) -> str:
        """
        Translate using LLM.
        
        Args:
            theorem: The theorem statement
            proof: The proof text
            
        Returns:
            The translated proof
        """
        logger.info("Using LLM-based translation")
        
        try:
            # Try to use the LLM translator
            from translation.llm_translator import translate_proof_with_llm
            return translate_proof_with_llm(theorem, proof, self.target_prover)
        except ImportError:
            # Fall back to OpenAI if available
            try:
                from llm.openai_client import translate_proof_with_openai
                return translate_proof_with_openai(theorem, proof, self.target_prover)
            except ImportError:
                logger.error("LLM translation modules not available")
                raise
    
    def _translate_with_rules(self, theorem: str, proof: str) -> str:
        """
        Translate using rule-based system.
        
        Args:
            theorem: The theorem statement
            proof: The proof text
            
        Returns:
            The translated proof
        """
        logger.info("Using rule-based translation")
        
        # Parse the proof
        from nlp.proof_parser import parse_math_proof
        parsed_info = parse_math_proof(proof)
        
        # Build IR
        from ir.proof_builder import ProofBuilder
        proof_builder = ProofBuilder()
        proof_ir = proof_builder.build_proof_ir(
            parsed_statements=parsed_info["parsed_statements"],
            proof_structure=parsed_info["proof_structure"],
            original_theorem=theorem,
            original_proof=proof
        )
        
        # Use the backend to translate
        return self.backend.translate(proof_ir)


# Standalone functions for use in other modules

def translate_proof(theorem: str, proof: str, target_prover: str = "coq", use_llm: bool = False) -> Dict[str, Any]:
    """
    Translate a proof using the translation system.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        target_prover: The target theorem prover
        use_llm: Whether to use LLM assistance
        
    Returns:
        Dictionary with translation results
    """
    translator = ProofTranslator(target_prover=target_prover, use_llm=use_llm)
    return translator.translate(theorem, proof)