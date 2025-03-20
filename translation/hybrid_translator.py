"""
Hybrid translator that combines NaturalProofs with rule-based translation.
"""

import logging
from typing import Dict, Any, Optional

from backends.backend_interface import BackendRegistry
from knowledge_base.simple_kb import SimpleKnowledgeBase
from verification.feedback_loop import process_verification_result, apply_feedback_fixes
from core.understanding.mathematical_parser import MathematicalParser
from nlp import parse_mathematical_proof

logger = logging.getLogger(__name__)

class HybridTranslator:
    """Hybrid translator that combines NaturalProofs with our backend."""
    
    def __init__(self, target_prover: str = "coq", model_path: Optional[str] = None, use_llm: bool = False):
        """
        Initialize the hybrid translator.
        
        Args:
            target_prover: The target theorem prover ("coq" or "lean")
            model_path: Optional path to a pre-trained model
            use_llm: Whether to use LLM assistance for translation
        """
        self.kb = SimpleKnowledgeBase()
        
        # Try to initialize the NaturalProofs parser
        try:
            self.np_parser = MathematicalParser(model_path)
            self.use_np = True
            logger.info(f"Initialized NaturalProofs parser for {target_prover}")
        except Exception as e:
            logger.warning(f"Could not initialize NaturalProofs parser: {e}. Using fallback parser.")
            self.use_np = False
        
        # Initialize the backend
        self.backend = BackendRegistry.get_backend(target_prover)
        self.target_prover = target_prover
        self.use_llm = use_llm
        
        logger.info(f"Initialized hybrid translator for {target_prover}")
    
    def translate(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Translate an informal proof to a formal proof.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with translation results
        """
        # Step 1: Parse and understand the proof
        logger.info("Parsing and understanding the proof")
        
        if self.use_np:
            # Use NaturalProofs parser
            try:
                proof_ir = self.np_parser.parse_and_convert(theorem_text, proof_text)
                logger.info("Successfully used NaturalProofs parser")
            except Exception as e:
                logger.warning(f"NaturalProofs parsing failed: {e}. Falling back to standard parser.")
                parsed_proof = parse_mathematical_proof(f"{theorem_text}\n\nProof: {proof_text}")
                proof_ir = parsed_proof["proof_ir"]
        else:
            # Use standard parser
            parsed_proof = parse_mathematical_proof(f"{theorem_text}\n\nProof: {proof_text}")
            proof_ir = parsed_proof["proof_ir"]
        
        # Step 2: Enhance IR with domain-specific knowledge
        logger.info("Enhancing with domain-specific knowledge")
        self._enhance_with_domain_knowledge(proof_ir)
        
        # Step 3: Generate formal proof using our backend
        logger.info(f"Generating formal proof using {self.target_prover} backend")
        formal_proof = self.backend.translate(proof_ir)
        
        # Step 4: Verify and refine if needed
        verified = False
        error_message = None
        
        if self.backend.is_installed():
            logger.info("Verifying the proof")
            verified, error_message = self.backend.verify(formal_proof)
            
            # Step 5: Apply feedback if verification failed
            if not verified:
                logger.info("Verification failed, applying feedback")
                
                # Get feedback from verification result
                feedback = process_verification_result(
                    proof_ir, formal_proof, verified, error_message, self.target_prover
                )
                
                # Apply fixes based on feedback
                if "feedback" in feedback:
                    logger.info("Applying feedback fixes")
                    fixed_proof = apply_feedback_fixes(formal_proof, feedback["feedback"])
                    
                    # Verify again
                    reverify, reverify_error = self.backend.verify(fixed_proof)
                    if reverify:
                        logger.info("Verification succeeded after applying fixes")
                        formal_proof = fixed_proof
                        verified = True
                        error_message = None
                    else:
                        logger.info("Verification still failed after applying fixes")
                
                # Try LLM if available and verification still failed
                if not verified and self.use_llm:
                    logger.info("Using LLM to fix proof")
                    llm_fixed_proof = self._fix_with_llm(formal_proof, error_message)
                    
                    # Verify the LLM-fixed proof
                    llm_verified, llm_error = self.backend.verify(llm_fixed_proof)
                    if llm_verified:
                        logger.info("LLM fix succeeded")
                        formal_proof = llm_fixed_proof
                        verified = True
                        error_message = None
        else:
            logger.warning(f"{self.target_prover} backend is not installed, skipping verification")
            error_message = f"{self.target_prover} backend not installed"
        
        return {
            "formal_proof": formal_proof,
            "verified": verified,
            "error_message": error_message,
            "domain": proof_ir.domain.get("primary_domain", ""),
            "pattern": proof_ir.pattern.get("name", "")
        }
    
    def _enhance_with_domain_knowledge(self, proof_ir):
        """
        Enhance the IR with domain-specific knowledge.
        
        Args:
            proof_ir: The proof intermediate representation
        """
        domain = proof_ir.domain.get("primary_domain", "")
        pattern = proof_ir.pattern.get("name", "")
        
        logger.debug(f"Enhancing IR with domain: {domain}, pattern: {pattern}")
        
        # Initialize variable types tracking if not present
        if "variable_types" not in proof_ir.metadata:
            proof_ir.metadata["variable_types"] = {}
        
        # Determine variable types based on pattern and domain
        variables = proof_ir.metadata.get("variables", [])
        
        # Default type for variables is nat (natural numbers) for most mathematical proofs
        for var in variables:
            if var not in proof_ir.metadata["variable_types"]:
                proof_ir.metadata["variable_types"][var] = "nat"
        
        # Check if we need to use Z (integers) instead based on the proof content
        for node in proof_ir.proof_tree:
            content = str(node.content).lower()
            if "integer" in content or "z" in content and not "zero" in content:
                # Update variable types to Z for integer proofs
                for var in variables:
                    proof_ir.metadata["variable_types"][var] = "Z"
        
        # Add domain-specific library dependencies
        for var in variables:
            var_type = proof_ir.metadata["variable_types"].get(var, "nat")
            libraries = self.kb.get_libraries_for_concept(var_type, domain, self.target_prover)
            for library in libraries:
                logger.debug(f"Adding library dependency: {library} for variable {var} of type {var_type}")
                proof_ir.add_library_dependency(library, f"Required for {var} of type {var_type}", [var])
        
        # Add domain-specific libraries
        domain_libraries = self.kb.get_domain_libraries(domain, self.target_prover)
        for library in domain_libraries:
            logger.debug(f"Adding domain library: {library}")
            proof_ir.add_library_dependency(library, f"Domain library for {domain}", [])
        
        # Add pattern-specific tactics
        tactics = self.kb.get_tactics_for_pattern(pattern, self.target_prover)
        if tactics:
            logger.debug(f"Adding {len(tactics)} pattern-specific tactics")
            proof_ir.tactics.extend(tactics)
    
    def _fix_with_llm(self, formal_proof: str, error_message: Optional[str]) -> str:
        """
        Attempt to fix the proof using LLM if enabled.
        
        Args:
            formal_proof: The formal proof to fix
            error_message: The error message from verification
            
        Returns:
            Fixed formal proof
        """
        if not self.use_llm:
            return formal_proof
        
        try:
            # Here we would integrate with an LLM to fix the proof
            # For now, we'll just return the original proof
            logger.info("LLM fixing is enabled but not yet implemented")
            return formal_proof
        except Exception as e:
            logger.error(f"Error in LLM fixing: {e}")
            return formal_proof
    
    def translate_with_rules(self, proof_ir):
        """
        Translate using only rule-based translation (no NaturalProofs or LLM).
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The translated formal proof
        """
        # Enhance with domain knowledge
        self._enhance_with_domain_knowledge(proof_ir)
        
        # Translate with backend
        return self.backend.translate(proof_ir)