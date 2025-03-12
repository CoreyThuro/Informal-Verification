"""
Domain-specific translation strategies.
Provides specialized strategies for different mathematical domains.
"""

import logging
from typing import Dict, List, Any, Optional

from ir.proof_ir import ProofIR, TacticType, TacticInfo

# Configure logging
logger = logging.getLogger("domain_strategies")

class TranslationStrategy:
    """Base class for translation strategies."""
    
    def __init__(self, kb=None):
        """
        Initialize the strategy.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply the strategy to a proof IR.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        # Default implementation
        return {
            "strategy": "direct",
            "libraries": [],
            "tactics": [],
            "parameters": {
                "use_automation": True
            }
        }

class NumberTheoryStrategy(TranslationStrategy):
    """Strategy for number theory proofs."""
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply number theory-specific translation strategy.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        # Get domain libraries
        libraries = self._get_domain_libraries(target_prover)
        
        # Get tactics
        tactics = self._generate_tactics(proof_ir, target_prover)
        
        # Get pattern-specific enhancements
        pattern = proof_ir.pattern.get("name", "")
        
        if pattern == "induction":
            tactics = self._enhance_induction_tactics(tactics, proof_ir, target_prover)
        elif pattern == "contradiction":
            tactics = self._enhance_contradiction_tactics(tactics, proof_ir, target_prover)
        elif pattern == "evenness_proof":
            tactics = self._enhance_evenness_tactics(tactics, proof_ir, target_prover)
        
        # Set up parameters
        parameters = {
            "use_automation": True,
            "prefer_libraries": self._get_preferred_libraries(target_prover)
        }
        
        return {
            "strategy": "number_theory",
            "libraries": libraries,
            "tactics": tactics,
            "parameters": parameters
        }
    
    def _get_domain_libraries(self, target_prover: str) -> List[str]:
        """
        Get required libraries for number theory.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of library imports
        """
        if self.kb:
            return self.kb.get_domain_libraries("11", target_prover)
        
        # Default libraries if no knowledge base
        if target_prover.lower() == "coq":
            return ["Arith", "ZArith", "Lia"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Data.Nat.Basic", "Mathlib.Data.Nat.Parity", "Mathlib.Tactic.Ring"]
        else:
            return []
    
    def _get_preferred_libraries(self, target_prover: str) -> List[str]:
        """
        Get preferred libraries for number theory.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of preferred libraries
        """
        if target_prover.lower() == "coq":
            return ["Arith", "ZArith", "Lia"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Data.Nat.Basic", "Mathlib.Data.Nat.Parity", "Mathlib.Tactic.Ring"]
        else:
            return []
    
    def _generate_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Generate number theory tactics.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Get variables
        variables = proof_ir.metadata.get("variables", [])
        
        # Add basic tactics
        tactics.append({
            "tactic_type": TacticType.INTRO.value,
            "arguments": variables,
            "description": f"Introduce variables: {', '.join(variables)}"
        })
        
        return tactics
    
    def _enhance_induction_tactics(self, tactics: List[Dict[str, Any]], 
                                  proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for induction proofs in number theory.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Find induction variable
        induction_var = self._find_induction_variable(proof_ir)
        
        # Add induction tactic
        tactics.append({
            "tactic_type": TacticType.INDUCTION.value,
            "arguments": [induction_var],
            "description": f"Apply induction on {induction_var}"
        })
        
        # Add simplification for base case
        tactics.append({
            "tactic_type": TacticType.SIMPLIFY.value,
            "arguments": [],
            "description": "Simplify expressions in base case"
        })
        
        # Add number theory automation tactics
        if target_prover.lower() == "coq":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["lia"],
                "description": "Use linear integer arithmetic solver"
            })
        elif target_prover.lower() == "lean":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["linarith"],
                "description": "Use linear arithmetic solver"
            })
        
        return tactics
    
    def _enhance_contradiction_tactics(self, tactics: List[Dict[str, Any]], 
                                      proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for contradiction proofs in number theory.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add contradiction tactic
        tactics.append({
            "tactic_type": TacticType.CONTRADICTION.value,
            "arguments": [],
            "description": "Set up proof by contradiction"
        })
        
        # Add number theory automation
        if target_prover.lower() == "coq":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["nia"],
                "description": "Use non-linear integer arithmetic"
            })
        elif target_prover.lower() == "lean":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["nlinarith"],
                "description": "Use non-linear arithmetic"
            })
        
        return tactics
    
    def _enhance_evenness_tactics(self, tactics: List[Dict[str, Any]], 
                                 proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for evenness proofs in number theory.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Find the variable
        variables = proof_ir.metadata.get("variables", [])
        var = variables[0] if variables else "n"
        
        # Add exists tactic for providing the witness
        tactics.append({
            "tactic_type": TacticType.EXISTS.value,
            "arguments": [var],
            "description": f"Provide witness {var} for evenness"
        })
        
        # Add ring tactic for algebraic simplification
        tactics.append({
            "tactic_type": TacticType.CUSTOM.value,
            "arguments": ["ring"],
            "description": "Solve with ring arithmetic"
        })
        
        return tactics
    
    def _find_induction_variable(self, proof_ir: ProofIR) -> str:
        """
        Find the induction variable in a proof.
        
        Args:
            proof_ir: The proof IR
            
        Returns:
            The induction variable
        """
        # Check if there's an induction tactic specified
        for tactic in proof_ir.tactics:
            if tactic.tactic_type == TacticType.INDUCTION and tactic.arguments:
                return tactic.arguments[0]
        
        # Fall back to variables in the proof
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            # Prefer common induction variables
            common_vars = ["n", "m", "k", "i", "j"]
            for var in common_vars:
                if var in variables:
                    return var
            
            # Otherwise, use the first variable
            return variables[0]
        
        # Default
        return "n"

class AlgebraStrategy(TranslationStrategy):
    """Strategy for algebraic proofs."""
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply algebra-specific translation strategy.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        # Get domain libraries
        libraries = self._get_domain_libraries(target_prover)
        
        # Get tactics
        tactics = self._generate_tactics(proof_ir, target_prover)
        
        # Get pattern-specific enhancements
        pattern = proof_ir.pattern.get("name", "")
        
        if pattern == "direct":
            tactics = self._enhance_direct_tactics(tactics, proof_ir, target_prover)
        elif pattern == "case_analysis":
            tactics = self._enhance_case_tactics(tactics, proof_ir, target_prover)
        
        # Set up parameters
        parameters = {
            "use_automation": True,
            "prefer_libraries": self._get_preferred_libraries(target_prover)
        }
        
        return {
            "strategy": "algebra",
            "libraries": libraries,
            "tactics": tactics,
            "parameters": parameters
        }
    
    def _get_domain_libraries(self, target_prover: str) -> List[str]:
        """
        Get required libraries for algebra.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of library imports
        """
        if self.kb:
            return self.kb.get_domain_libraries("12-20", target_prover)
        
        # Default libraries if no knowledge base
        if target_prover.lower() == "coq":
            return ["Algebra", "Ring", "Field", "LinearAlgebra"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Algebra.Ring.Basic", "Mathlib.LinearAlgebra.Basic", "Mathlib.Tactic.Ring"]
        else:
            return []
    
    def _get_preferred_libraries(self, target_prover: str) -> List[str]:
        """
        Get preferred libraries for algebra.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of preferred libraries
        """
        if target_prover.lower() == "coq":
            return ["Ring", "Field", "LinearAlgebra"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Algebra.Ring.Basic", "Mathlib.Tactic.Ring"]
        else:
            return []
    
    def _generate_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Generate algebra tactics.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Get variables
        variables = proof_ir.metadata.get("variables", [])
        
        # Add basic tactics
        tactics.append({
            "tactic_type": TacticType.INTRO.value,
            "arguments": variables,
            "description": f"Introduce variables: {', '.join(variables)}"
        })
        
        return tactics
    
    def _enhance_direct_tactics(self, tactics: List[Dict[str, Any]], 
                               proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for direct proofs in algebra.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add ring tactic for algebraic simplification
        tactics.append({
            "tactic_type": TacticType.CUSTOM.value,
            "arguments": ["ring"],
            "description": "Apply ring tactic for algebraic manipulation"
        })
        
        # Add field tactic if needed
        if self._has_division(proof_ir):
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["field"],
                "description": "Apply field tactic for rational expressions"
            })
        
        return tactics
    
    def _enhance_case_tactics(self, tactics: List[Dict[str, Any]], 
                             proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for case analysis proofs in algebra.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Find the variable for case analysis
        case_var = self._find_case_variable(proof_ir)
        
        # Add case analysis tactic
        tactics.append({
            "tactic_type": TacticType.CASE_ANALYSIS.value,
            "arguments": [case_var],
            "description": f"Case analysis on {case_var}"
        })
        
        # Add simplification for each case
        tactics.append({
            "tactic_type": TacticType.SIMPLIFY.value,
            "arguments": [],
            "description": "Simplify expressions in each case"
        })
        
        return tactics
    
    def _has_division(self, proof_ir: ProofIR) -> bool:
        """
        Check if the proof involves division.
        
        Args:
            proof_ir: The proof IR
            
        Returns:
            True if division is present, False otherwise
        """
        # Check original text for division symbols
        original_proof = proof_ir.original_proof or ""
        original_theorem = proof_ir.original_theorem or ""
        
        if "/" in original_proof or "/" in original_theorem:
            return True
        
        if "\\frac" in original_proof or "\\frac" in original_theorem:
            return True
        
        if "divided by" in original_proof.lower() or "divided by" in original_theorem.lower():
            return True
        
        return False
    
    def _find_case_variable(self, proof_ir: ProofIR) -> str:
        """
        Find the variable for case analysis.
        
        Args:
            proof_ir: The proof IR
            
        Returns:
            The case variable
        """
        # Check if there's a case analysis tactic specified
        for tactic in proof_ir.tactics:
            if tactic.tactic_type == TacticType.CASE_ANALYSIS and tactic.arguments:
                return tactic.arguments[0]
        
        # Fall back to variables in the proof
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            return variables[0]
        
        # Default
        return "h"

class AnalysisStrategy(TranslationStrategy):
    """Strategy for analysis proofs."""
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply analysis-specific translation strategy.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        # Get domain libraries
        libraries = self._get_domain_libraries(target_prover)
        
        # Get tactics
        tactics = self._generate_tactics(proof_ir, target_prover)
        
        # Get pattern-specific enhancements
        pattern = proof_ir.pattern.get("name", "")
        
        if pattern == "epsilon_delta":
            tactics = self._enhance_epsilon_delta_tactics(tactics, proof_ir, target_prover)
        elif pattern == "contradiction":
            tactics = self._enhance_contradiction_tactics(tactics, proof_ir, target_prover)
        
        # Set up parameters
        parameters = {
            "use_automation": True,
            "prefer_libraries": self._get_preferred_libraries(target_prover)
        }
        
        return {
            "strategy": "analysis",
            "libraries": libraries,
            "tactics": tactics,
            "parameters": parameters
        }
    
    def _get_domain_libraries(self, target_prover: str) -> List[str]:
        """
        Get required libraries for analysis.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of library imports
        """
        if self.kb:
            return self.kb.get_domain_libraries("26-42", target_prover)
        
        # Default libraries if no knowledge base
        if target_prover.lower() == "coq":
            return ["Reals", "Ranalysis", "Rtrigo", "RiemannInt"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Analysis.RealFunction", "Mathlib.Analysis.Calculus.Deriv.Basic", "Mathlib.Analysis.Calculus.FDeriv.Basic"]
        else:
            return []
    
    def _get_preferred_libraries(self, target_prover: str) -> List[str]:
        """
        Get preferred libraries for analysis.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of preferred libraries
        """
        if target_prover.lower() == "coq":
            return ["Reals", "Ranalysis"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Analysis.RealFunction", "Mathlib.Analysis.Calculus.Deriv.Basic"]
        else:
            return []
    
    def _generate_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Generate analysis tactics.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Get variables
        variables = proof_ir.metadata.get("variables", [])
        
        # Add basic tactics
        tactics.append({
            "tactic_type": TacticType.INTRO.value,
            "arguments": variables,
            "description": f"Introduce variables: {', '.join(variables)}"
        })
        
        return tactics
    
    def _enhance_epsilon_delta_tactics(self, tactics: List[Dict[str, Any]], 
                                      proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for epsilon-delta proofs in analysis.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add epsilon intro
        tactics.append({
            "tactic_type": TacticType.INTRO.value,
            "arguments": ["epsilon", "H_epsilon"],
            "description": "Introduce epsilon and positivity hypothesis"
        })
        
        # Add delta existence
        tactics.append({
            "tactic_type": TacticType.EXISTS.value,
            "arguments": ["delta"],
            "description": "Provide delta witness"
        })
        
        # Add field tactic for real arithmetic
        tactics.append({
            "tactic_type": TacticType.CUSTOM.value,
            "arguments": ["field"],
            "description": "Apply field tactic for real arithmetic"
        })
        
        return tactics
    
    def _enhance_contradiction_tactics(self, tactics: List[Dict[str, Any]], 
                                      proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for contradiction proofs in analysis.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add contradiction tactic
        tactics.append({
            "tactic_type": TacticType.CONTRADICTION.value,
            "arguments": [],
            "description": "Set up proof by contradiction"
        })
        
        # Add real number tactics
        if target_prover.lower() == "coq":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["auto with real"],
                "description": "Apply real number automation"
            })
        elif target_prover.lower() == "lean":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["exact"],
                "description": "Use real number facts"
            })
        
        return tactics

class TopologyStrategy(TranslationStrategy):
    """Strategy for topology proofs."""
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply topology-specific translation strategy.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        # Get domain libraries
        libraries = self._get_domain_libraries(target_prover)
        
        # Get tactics
        tactics = self._generate_tactics(proof_ir, target_prover)
        
        # Get pattern-specific enhancements
        pattern = proof_ir.pattern.get("name", "")
        
        if pattern == "direct":
            tactics = self._enhance_direct_tactics(tactics, proof_ir, target_prover)
        elif pattern == "contradiction":
            tactics = self._enhance_contradiction_tactics(tactics, proof_ir, target_prover)
        
        # Set up parameters
        parameters = {
            "use_automation": True,
            "prefer_libraries": self._get_preferred_libraries(target_prover)
        }
        
        return {
            "strategy": "topology",
            "libraries": libraries,
            "tactics": tactics,
            "parameters": parameters
        }
    
    def _get_domain_libraries(self, target_prover: str) -> List[str]:
        """
        Get required libraries for topology.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of library imports
        """
        if self.kb:
            return self.kb.get_domain_libraries("54-55", target_prover)
        
        # Default libraries if no knowledge base
        if target_prover.lower() == "coq":
            return ["Topology", "Reals", "MetricSpaces"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Topology.Basic", "Mathlib.Topology.MetricSpace.Basic", "Mathlib.Topology.Connected"]
        else:
            return []
    
    def _get_preferred_libraries(self, target_prover: str) -> List[str]:
        """
        Get preferred libraries for topology.
        
        Args:
            target_prover: The target prover
            
        Returns:
            List of preferred libraries
        """
        if target_prover.lower() == "coq":
            return ["Topology", "MetricSpaces"]
        elif target_prover.lower() == "lean":
            return ["Mathlib.Topology.Basic", "Mathlib.Topology.MetricSpace.Basic"]
        else:
            return []
    
    def _generate_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Generate topology tactics.
        
        Args:
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Get variables
        variables = proof_ir.metadata.get("variables", [])
        
        # Add basic tactics
        tactics.append({
            "tactic_type": TacticType.INTRO.value,
            "arguments": variables,
            "description": f"Introduce variables: {', '.join(variables)}"
        })
        
        return tactics
    
    def _enhance_direct_tactics(self, tactics: List[Dict[str, Any]], 
                               proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for direct proofs in topology.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add unfold tactic for definitions
        tactics.append({
            "tactic_type": TacticType.CUSTOM.value,
            "arguments": ["unfold"],
            "description": "Unfold topological definitions"
        })
        
        # Add automation
        if target_prover.lower() == "coq":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["auto with topology"],
                "description": "Apply topological automation"
            })
        elif target_prover.lower() == "lean":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["exact"],
                "description": "Use topological facts"
            })
        
        return tactics
    
    def _enhance_contradiction_tactics(self, tactics: List[Dict[str, Any]], 
                                      proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance tactics for contradiction proofs in topology.
        
        Args:
            tactics: Current tactics
            proof_ir: The proof IR
            target_prover: The target prover
            
        Returns:
            Enhanced tactics
        """
        # Add contradiction tactic
        tactics.append({
            "tactic_type": TacticType.CONTRADICTION.value,
            "arguments": [],
            "description": "Set up proof by contradiction"
        })
        
        # Add topological tactics
        if target_prover.lower() == "coq":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["auto with topology"],
                "description": "Apply topological automation"
            })
        elif target_prover.lower() == "lean":
            tactics.append({
                "tactic_type": TacticType.CUSTOM.value,
                "arguments": ["exact"],
                "description": "Use topological facts"
            })
        
        return tactics

def get_strategy_for_domain(domain: str, kb=None) -> TranslationStrategy:
    """
    Get a translation strategy for a domain.
    
    Args:
        domain: The MSC domain code
        kb: Optional knowledge base
        
    Returns:
        A translation strategy for the domain
    """
    # Map domain codes to strategies
    if domain in ["11"]:  # Number Theory
        return NumberTheoryStrategy(kb)
    elif domain in ["12", "13", "14", "15", "16", "17", "18", "19", "20"]:  # Algebra
        return AlgebraStrategy(kb)
    elif domain in ["26", "27", "28", "30", "31", "32", "33", "34", "35", "40", "41", "42"]:  # Analysis
        return AnalysisStrategy(kb)
    elif domain in ["54", "55"]:  # Topology
        return TopologyStrategy(kb)
    else:
        # Default strategy
        return TranslationStrategy(kb)