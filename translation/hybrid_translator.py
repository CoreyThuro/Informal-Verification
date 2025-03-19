# translation/hybrid_translator.py
import logging
from typing import Dict, Any, Optional

from core.understanding.mathematical_parser import MathematicalParser
from backends.backend_interface import BackendRegistry
from knowledge_base.simple_kb import SimpleKnowledgeBase
from verification.feedback_loop import process_verification_result, apply_feedback_fixes

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
        self.parser = MathematicalParser(model_path)
        self.kb = SimpleKnowledgeBase()
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
        parsed_proof = self.parser.parse_proof(theorem_text, proof_text)
        
        # Step 2: Convert to our IR format
        logger.info("Converting to IR format")
        proof_ir = self.parser.convert_to_ir(parsed_proof)
        
        # Step 3: Enhance IR with domain-specific knowledge
        logger.info("Enhancing with domain-specific knowledge")
        self._enhance_with_domain_knowledge(proof_ir)
        
        # Step 4: Generate formal proof using our backend
        logger.info(f"Generating formal proof using {self.target_prover} backend")
        formal_proof = self.backend.translate(proof_ir)
        
        # Step 5: Verify and refine if needed
        verified = False
        error_message = None
        
        if self.backend.is_installed():
            logger.info("Verifying the proof")
            verified, error_message = self.backend.verify(formal_proof)
            
            # Step 6: Apply feedback if verification failed
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
            "domain": parsed_proof["domain"],
            "pattern": parsed_proof["pattern"]["name"]
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
        
        # Add domain-specific library dependencies
        for var in proof_ir.metadata.get("variables", []):
            libraries = self.kb.get_libraries_for_concept(var, domain, self.target_prover)
            for library in libraries:
                logger.debug(f"Adding library dependency: {library} for variable {var}")
                proof_ir.add_library_dependency(library, f"Required for {var}", [var])
        
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