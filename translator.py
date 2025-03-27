"""
Enhanced translator module for proof translation.
"""

from typing import Dict, Any, List, Optional
import re

from patterns.recognizer import recognize_pattern
from patterns.enhanced_recognizer import enhanced_recognize_pattern
from patterns.translators.evenness import translate_evenness_proof
from patterns.translators.induction import translate_induction_proof
from patterns.translators.contradiction import translate_contradiction_proof
from patterns.translators.cases import translate_cases_proof
from patterns.translators.direct import translate_direct_proof
from coq.verifier import verify_coq_proof
from coq.feedback import apply_feedback
from knowledge.kb import KnowledgeBase

class ProofTranslator:
    """
    Main translator class for converting informal proofs to Coq.
    """
    
    def __init__(self):
        """Initialize the translator."""
        self.kb = KnowledgeBase()
    
    def translate(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Translate an informal proof to a formal Coq proof.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with translation results
        """
        # Detect pattern and domain using enhanced recognizer with fallback
        try:
            pattern, pattern_info = enhanced_recognize_pattern(theorem_text, proof_text)
            print(f"Using enhanced pattern recognition: {pattern}")
        except Exception as e:
            # Fallback to basic recognizer if enhanced one fails
            print(f"Enhanced recognizer failed: {e}, falling back to basic recognizer")
            pattern, pattern_info = recognize_pattern(theorem_text, proof_text)
        
        domain = self._detect_domain(theorem_text, proof_text)
        
        print(f"Detected pattern: {pattern}, domain: {domain}")
        
        # Generate formal proof based on pattern
        if pattern == "evenness":
            variable = pattern_info.get("variable", "n")
            # Pass additional context to the enhanced translator
            formal_proof = translate_evenness_proof(
                variable, 
                domain, 
                theorem_text, 
                proof_text, 
                pattern_info.get("structure_info", None)
            )
        elif pattern == "induction":
            variable = pattern_info.get("variable", "n")
            formal_proof = translate_induction_proof(variable, theorem_text, proof_text, domain)
        elif pattern == "contradiction":
            variables = pattern_info.get("variables", ["n"])
            formal_proof = translate_contradiction_proof(theorem_text, proof_text, variables, domain)
        elif pattern == "cases":
            variables = pattern_info.get("variables", ["n"])
            case_var = pattern_info.get("case_var", variables[0] if variables else "n")
            formal_proof = translate_cases_proof(theorem_text, proof_text, variables, case_var, domain)
        else:
            variables = pattern_info.get("variables", ["n"])
            formal_proof = translate_direct_proof(theorem_text, proof_text, variables, domain)
        
        # Verify the proof
        verified, error = verify_coq_proof(formal_proof)
        
        # Apply feedback if verification failed
        if not verified and error:
            fixed_proof = apply_feedback(formal_proof, error)
            reverified, new_error = verify_coq_proof(fixed_proof)
            
            if reverified:
                formal_proof = fixed_proof
                verified = True
                error = None
            else:
                # Try additional domain-specific fixes
                domain_fixed_proof = self._apply_domain_fixes(fixed_proof, new_error, domain)
                if domain_fixed_proof != fixed_proof:
                    domain_verified, domain_error = verify_coq_proof(domain_fixed_proof)
                    if domain_verified:
                        formal_proof = domain_fixed_proof
                        verified = True
                        error = None
                    else:
                        error = domain_error
        
        return {
            "formal_proof": formal_proof,
            "verified": verified,
            "error_message": error,
            "pattern": pattern,
            "domain": domain
        }
    
    def _detect_domain(self, theorem_text: str, proof_text: str) -> str:
        """
        Detect the mathematical domain with improved semantic analysis.
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Domain code (e.g., "11" for number theory)
        """
        combined = f"{theorem_text} {proof_text}".lower()
        
        # Get domain keywords from knowledge base
        domain_keywords = self.kb.get_domain_keywords() if hasattr(self.kb, 'get_domain_keywords') else {
            "11": ["prime", "number", "integer", "divisible", "even", "odd", "gcd", 
                   "divisor", "factor", "multiple", "remainder", "modulo", "congruent"],
            "12-20": ["group", "ring", "field", "algebra", "vector", "matrix", "linear", 
                      "polynomial", "homomorphism", "isomorphism", "commutative", "associative"],
            "26-42": ["limit", "continuous", "derivative", "integral", "function", "series", 
                      "sequence", "convergent", "divergent", "differentiable", "bounded"],
            "54-55": ["topology", "open", "closed", "compact", "connected", "neighborhood", 
                      "homeomorphic", "metric", "space", "continuous"]
        }
        
        # Weighted scoring for domain detection
        scores = {domain: 0 for domain in domain_keywords}
        
        # Check for exact keyword matches with context awareness
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                # Simple presence check
                if keyword in combined:
                    scores[domain] += 1
                    
                # Check for phrases that strongly indicate a domain
                strong_indicators = {
                    "11": ["divisible by", "factor of", "prime number", "greatest common divisor"],
                    "12-20": ["abelian group", "vector space", "linear algebra", "field extension"],
                    "26-42": ["continuous function", "derivative of", "converges to", "bounded sequence"],
                    "54-55": ["open set", "closed set", "compact space", "connected component"]
                }
                
                if domain in strong_indicators:
                    for indicator in strong_indicators[domain]:
                        if indicator in combined:
                            scores[domain] += 3  # Higher weight for strong indicators
        
        # Check for mathematical symbols that indicate domains
        symbols = {
            "11": [r"\bdiv\b", r"\bmod\b", r"\bgcd\b", r"\|", r"≡", r"≢"],
            "12-20": [r"⊕", r"⊗", r"⊆", r"⊇", r"∈", r"∉", r"∀", r"∃"],
            "26-42": [r"∫", r"∂", r"∑", r"lim", r"→", r"∞"],
            "54-55": [r"∩", r"∪", r"⊂", r"⊃", r"∅"]
        }
        
        for domain, domain_symbols in symbols.items():
            for symbol in domain_symbols:
                if re.search(symbol, combined):
                    scores[domain] += 2
        
        # Return domain with highest score, or default
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "00"  # General mathematics
    
    def _apply_domain_fixes(self, proof_script: str, error_message: str, domain: str) -> str:
        """
        Apply domain-specific fixes to the proof.
        
        Args:
            proof_script: The proof script
            error_message: The error message
            domain: The mathematical domain
            
        Returns:
            Fixed proof script
        """
        # Add domain-specific imports if needed
        imports = self.kb.get_imports_for_domain(domain)
        
        # Check if imports are already in the script
        for import_stmt in imports:
            if import_stmt not in proof_script:
                proof_script = import_stmt + "\n" + proof_script
        
        # Apply domain-specific fixes
        if domain == "11":  # Number theory
            # Add ring tactic if missing
            if "ring" not in proof_script and "Error: The reference ring" in error_message:
                proof_script = re.sub(r'Proof\.', 'Proof.\n  ring.', proof_script)
        
        return proof_script