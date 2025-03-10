"""
Tactic generator for theorem provers.
Generates appropriate proof tactics based on proof characteristics.
"""

from typing import Dict, List, Any, Optional, Union
import re

from ir.proof_ir import ProofIR, ProofNode, NodeType, TacticInfo, TacticType

class TacticGenerator:
    """
    Generates appropriate proof tactics for theorem provers.
    """
    
    def __init__(self, target_prover: str = "coq"):
        """
        Initialize the tactic generator.
        
        Args:
            target_prover: The target theorem prover
        """
        self.target_prover = target_prover.lower()
        
        # Initialize prover-specific tactic generators
        self.tactic_generators = {
            "coq": self._generate_coq_tactics,
            "lean": self._generate_lean_tactics
        }
    
    def generate_tactics(self, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate tactics for a proof.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of tactic information
        """
        # Get the appropriate tactic generator for the target prover
        if self.target_prover in self.tactic_generators:
            generator = self.tactic_generators[self.target_prover]
        else:
            # Default to Coq tactics
            generator = self._generate_coq_tactics
        
        # Generate tactics
        return generator(proof_ir)
    
    def _generate_coq_tactics(self, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Coq tactics for a proof.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of tactic information for Coq
        """
        tactics = []
        
        # Extract key information from the proof
        pattern_name = proof_ir.pattern.get("name", "direct")
        domain = proof_ir.domain.get("primary_domain", "")
        
        # Generate tactics based on pattern
        pattern_tactics = self._generate_pattern_tactics_coq(pattern_name, proof_ir)
        tactics.extend(pattern_tactics)
        
        # Generate tactics based on domain
        domain_tactics = self._generate_domain_tactics_coq(domain, proof_ir)
        tactics.extend(domain_tactics)
        
        # Add basic tactics that are almost always useful
        basic_tactics = self._generate_basic_tactics_coq(proof_ir)
        tactics.extend(basic_tactics)
        
        return tactics
    
    def _generate_lean_tactics(self, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Lean tactics for a proof.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of tactic information for Lean
        """
        tactics = []
        
        # Extract key information from the proof
        pattern_name = proof_ir.pattern.get("name", "direct")
        domain = proof_ir.domain.get("primary_domain", "")
        
        # Generate tactics based on pattern
        pattern_tactics = self._generate_pattern_tactics_lean(pattern_name, proof_ir)
        tactics.extend(pattern_tactics)
        
        # Generate tactics based on domain
        domain_tactics = self._generate_domain_tactics_lean(domain, proof_ir)
        tactics.extend(domain_tactics)
        
        # Add basic tactics that are almost always useful
        basic_tactics = self._generate_basic_tactics_lean(proof_ir)
        tactics.extend(basic_tactics)
        
        return tactics
    
    def _generate_pattern_tactics_coq(self, pattern: str, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Coq tactics based on proof pattern.
        
        Args:
            pattern: The proof pattern
            proof_ir: The proof intermediate representation
            
        Returns:
            List of pattern-specific tactics for Coq
        """
        tactics = []
        
        if pattern == "induction":
            # Find the variable to induct on
            induction_var = self._find_induction_variable(proof_ir)
            if induction_var:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.INDUCTION,
                    arguments=[induction_var],
                    description=f"Apply induction on {induction_var}"
                ))
            else:
                # Default to n if no variable found
                tactics.append(TacticInfo(
                    tactic_type=TacticType.INDUCTION,
                    arguments=["n"],
                    description="Apply induction on n"
                ))
        
        elif pattern == "contradiction":
            tactics.append(TacticInfo(
                tactic_type=TacticType.CONTRADICTION,
                description="Use proof by contradiction"
            ))
        
        elif pattern == "cases":
            # Find variable to use for case analysis
            case_var = self._find_case_variable(proof_ir)
            if case_var:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CASE_ANALYSIS,
                    arguments=[case_var],
                    description=f"Use case analysis on {case_var}"
                ))
            else:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CASE_ANALYSIS,
                    description="Use case analysis"
                ))
        
        elif pattern == "existence":
            # Find witness for existential
            witness = self._find_existential_witness(proof_ir)
            if witness:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.EXISTS,
                    arguments=[witness],
                    description=f"Provide witness {witness} for existential"
                ))
        
        return tactics
    
    def _generate_pattern_tactics_lean(self, pattern: str, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Lean tactics based on proof pattern.
        
        Args:
            pattern: The proof pattern
            proof_ir: The proof intermediate representation
            
        Returns:
            List of pattern-specific tactics for Lean
        """
        tactics = []
        
        if pattern == "induction":
            # Find the variable to induct on
            induction_var = self._find_induction_variable(proof_ir)
            if induction_var:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.INDUCTION,
                    arguments=[induction_var],
                    description=f"Apply induction on {induction_var}"
                ))
            else:
                # Default to n if no variable found
                tactics.append(TacticInfo(
                    tactic_type=TacticType.INDUCTION,
                    arguments=["n"],
                    description="Apply induction on n"
                ))
        
        elif pattern == "contradiction":
            tactics.append(TacticInfo(
                tactic_type=TacticType.CONTRADICTION,
                description="Use proof by contradiction"
            ))
        
        elif pattern == "cases":
            # Find variable to use for case analysis
            case_var = self._find_case_variable(proof_ir)
            if case_var:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CASE_ANALYSIS,
                    arguments=[case_var],
                    description=f"Use case analysis on {case_var}"
                ))
            else:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CASE_ANALYSIS,
                    description="Use case analysis"
                ))
        
        elif pattern == "existence":
            # Find witness for existential
            witness = self._find_existential_witness(proof_ir)
            if witness:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.EXISTS,
                    arguments=[witness],
                    description=f"Provide witness {witness} for existential"
                ))
        
        return tactics
    
    def _generate_domain_tactics_coq(self, domain: str, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Coq tactics based on mathematical domain.
        
        Args:
            domain: The mathematical domain
            proof_ir: The proof intermediate representation
            
        Returns:
            List of domain-specific tactics for Coq
        """
        tactics = []
        
        if domain in ["11", "12-20"]:  # Number theory or algebra
            # Check for arithmetic expressions
            has_arithmetic = False
            for node in proof_ir.proof_tree:
                if re.search(r'[+\-*/]', str(node.content)):
                    has_arithmetic = True
                    break
            
            if has_arithmetic:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CUSTOM,
                    arguments=["ring"],
                    description="Use ring for algebraic equalities"
                ))
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CUSTOM,
                    arguments=["lia"],
                    description="Use linear integer arithmetic"
                ))
        
        elif domain in ["26-42"]:  # Analysis
            tactics.append(TacticInfo(
                tactic_type=TacticType.CUSTOM,
                arguments=["field"],
                description="Use field for real-valued expressions"
            ))
        
        return tactics
    
    def _generate_domain_tactics_lean(self, domain: str, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate Lean tactics based on mathematical domain.
        
        Args:
            domain: The mathematical domain
            proof_ir: The proof intermediate representation
            
        Returns:
            List of domain-specific tactics for Lean
        """
        tactics = []
        
        if domain in ["11", "12-20"]:  # Number theory or algebra
            # Check for arithmetic expressions
            has_arithmetic = False
            for node in proof_ir.proof_tree:
                if re.search(r'[+\-*/]', str(node.content)):
                    has_arithmetic = True
                    break
            
            if has_arithmetic:
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CUSTOM,
                    arguments=["ring"],
                    description="Use ring for algebraic equalities"
                ))
                tactics.append(TacticInfo(
                    tactic_type=TacticType.CUSTOM,
                    arguments=["norm_num"],
                    description="Normalize numeric expressions"
                ))
        
        elif domain in ["26-42"]:  # Analysis
            tactics.append(TacticInfo(
                tactic_type=TacticType.CUSTOM,
                arguments=["field_simp"],
                description="Simplify field expressions"
            ))
        
        return tactics
    
    def _generate_basic_tactics_coq(self, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate basic Coq tactics applicable to most proofs.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of basic tactics for Coq
        """
        tactics = []
        
        # Find variables to introduce
        variables = proof_ir.metadata.get("variables", [])
        
        if variables:
            tactics.append(TacticInfo(
                tactic_type=TacticType.INTRO,
                arguments=variables,
                description=f"Introduce variables: {', '.join(variables)}"
            ))
        else:
            tactics.append(TacticInfo(
                tactic_type=TacticType.INTRO,
                description="Introduce variables"
            ))
        
        # Add simplification tactic
        tactics.append(TacticInfo(
            tactic_type=TacticType.SIMPLIFY,
            description="Simplify expressions"
        ))
        
        # Add auto tactic for closing simple goals
        tactics.append(TacticInfo(
            tactic_type=TacticType.AUTO,
            description="Attempt automatic proof"
        ))
        
        return tactics
    
    def _generate_basic_tactics_lean(self, proof_ir: ProofIR) -> List[TacticInfo]:
        """
        Generate basic Lean tactics applicable to most proofs.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of basic tactics for Lean
        """
        tactics = []
        
        # Find variables to introduce
        variables = proof_ir.metadata.get("variables", [])
        
        if variables:
            tactics.append(TacticInfo(
                tactic_type=TacticType.INTRO,
                arguments=variables,
                description=f"Introduce variables: {', '.join(variables)}"
            ))
        else:
            tactics.append(TacticInfo(
                tactic_type=TacticType.INTRO,
                description="Introduce variables"
            ))
        
        # Add simplification tactic
        tactics.append(TacticInfo(
            tactic_type=TacticType.SIMPLIFY,
            description="Simplify expressions"
        ))
        
        # Add auto tactic for closing simple goals
        tactics.append(TacticInfo(
            tactic_type=TacticType.AUTO,
            description="Attempt automatic proof"
        ))
        
        return tactics
    
    def _find_induction_variable(self, proof_ir: ProofIR) -> Optional[str]:
        """
        Find the variable to induct on.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The variable name or None if not found
        """
        # Check for induction_base nodes
        for node in proof_ir.proof_tree:
            if node.node_type == NodeType.INDUCTION_BASE:
                # Try to extract variable from text
                var_match = re.search(r'base\s+case\s+.*?([a-zA-Z][a-zA-Z0-9]*)\s*=\s*0', 
                                     str(node.content), re.IGNORECASE)
                if var_match:
                    return var_match.group(1)
        
        # Check for induction_step nodes
        for node in proof_ir.proof_tree:
            if node.node_type == NodeType.INDUCTION_STEP:
                var_match = re.search(r'([a-zA-Z][a-zA-Z0-9]*)\s*=\s*k\s*\+\s*1', 
                                     str(node.content), re.IGNORECASE)
                if var_match:
                    return var_match.group(1)
        
        # Look in original text for any variable used in induction context
        original_proof = proof_ir.original_proof
        if original_proof:
            var_match = re.search(r'induction\s+on\s+([a-zA-Z][a-zA-Z0-9]*)', 
                                 original_proof, re.IGNORECASE)
            if var_match:
                return var_match.group(1)
        
        # Fall back to variables mentioned in the proof
        variables = proof_ir.metadata.get("variables", [])
        
        # Common induction variables
        common_vars = ["n", "m", "k", "x"]
        
        # Look for common induction variables in the list
        for var in common_vars:
            if var in variables:
                return var
        
        # If no common variable found, return the first variable if any
        return variables[0] if variables else None
    
    def _find_case_variable(self, proof_ir: ProofIR) -> Optional[str]:
        """
        Find the variable to use for case analysis.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The variable name or None if not found
        """
        # Check for case nodes
        for node in proof_ir.proof_tree:
            if node.node_type == NodeType.CASE:
                # Try to extract variable from text
                var_match = re.search(r'case\s+(?:when\s+)?([a-zA-Z][a-zA-Z0-9]*)\s+is', 
                                     str(node.content), re.IGNORECASE)
                if var_match:
                    return var_match.group(1)
        
        # Look in original text for any variable used in case context
        original_proof = proof_ir.original_proof
        if original_proof:
            var_match = re.search(r'case(?:s)?\s+(?:on|for|of)\s+([a-zA-Z][a-zA-Z0-9]*)', 
                                 original_proof, re.IGNORECASE)
            if var_match:
                return var_match.group(1)
        
        # Fall back to variables mentioned in the proof
        variables = proof_ir.metadata.get("variables", [])
        
        # If no specific variable found, return the first variable if any
        return variables[0] if variables else None
    
    def _find_existential_witness(self, proof_ir: ProofIR) -> Optional[str]:
        """
        Find the witness for an existential proof.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            The witness expression or None if not found
        """
        # Check for exists_witness nodes
        for node in proof_ir.proof_tree:
            if node.node_type == NodeType.EXISTS_WITNESS:
                return str(node.content)
        
        # Look for typical patterns in the proof
        original_proof = proof_ir.original_proof
        if original_proof:
            # Look for "take x = ..." or "let x = ..." or "consider x = ..."
            witness_match = re.search(r'(?:take|let|consider)\s+.*?=\s*([^\.]+)', 
                                     original_proof, re.IGNORECASE)
            if witness_match:
                return witness_match.group(1).strip()
        
        return None


# Standalone functions for use in other modules

def generate_tactics(proof_ir: ProofIR, target_prover: str = "coq") -> List[TacticInfo]:
    """
    Generate tactics for a proof.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        
    Returns:
        List of tactic information
    """
    generator = TacticGenerator(target_prover)
    return generator.generate_tactics(proof_ir)

def get_suggested_tactics(proof_ir: ProofIR, target_prover: str = "coq") -> List[Dict[str, Any]]:
    """
    Get suggested tactics for a proof in a friendly format.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        
    Returns:
        List of tactic dictionaries
    """
    tactics = generate_tactics(proof_ir, target_prover)
    
    return [
        {
            "type": tactic.tactic_type.value,
            "arguments": tactic.arguments,
            "description": tactic.description
        }
        for tactic in tactics
    ]