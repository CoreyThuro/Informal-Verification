"""
Strategy selector for choosing translation approaches based on proof characteristics.
Determines optimal translation strategies for different theorem provers.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from ir.proof_ir import ProofIR, TacticInfo, TacticType
from nlp.domain_detector import detect_domain
from nlp.pattern_recognizer import recognize_pattern
from nlp.proof_parser import parse_proof

class TranslationStrategy(Enum):
    """Translation strategies for converting proofs to formal representations."""
    
    # Direct approach - convert statements one-by-one
    DIRECT = "direct"
    
    # Tactic-based approach - focus on proof tactics
    TACTIC_BASED = "tactic_based"
    
    # Term-based approach - construct proof terms directly
    TERM_BASED = "term_based"
    
    # Declarative approach - use declarative proof style (like Isar in Isabelle)
    DECLARATIVE = "declarative"
    
    # Structure-based approach - focus on logical structure
    STRUCTURE_BASED = "structure_based"
    
    # LLM-assisted approach - use language models to help translation
    LLM_ASSISTED = "llm_assisted"

class StrategySelector:
    """
    Selects an appropriate translation strategy based on proof characteristics.
    """
    
    def __init__(self, use_llm: bool = False, fallback_to_llm: bool = True):
        """
        Initialize the strategy selector.
        
        Args:
            use_llm: Whether to use language model assistance
            fallback_to_llm: Whether to fall back to LLM if other strategies fail
        """
        self.use_llm = use_llm
        self.fallback_to_llm = fallback_to_llm
        self.domain_strategy_mapping = self._initialize_domain_mapping()
        self.pattern_strategy_mapping = self._initialize_pattern_mapping()
    
    def select_strategy(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Select an appropriate translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Extract key information from IR
        domain = proof_ir.domain.get("primary_domain", "")
        pattern = proof_ir.pattern.get("name", "")
        
        # Consider domain-specific strategies
        domain_strategy = self._get_domain_strategy(domain, target_prover)
        
        # Consider pattern-specific strategies
        pattern_strategy = self._get_pattern_strategy(pattern, target_prover)
        
        # Consider prover-specific preferences
        prover_strategy = self._get_prover_strategy(target_prover)
        
        # Combine strategies with appropriate weights
        combined_strategy = self._combine_strategies(
            domain_strategy, pattern_strategy, prover_strategy,
            weights=(0.3, 0.5, 0.2)  # Pattern strategy has higher weight
        )
        
        # Consider LLM if enabled
        if self.use_llm:
            combined_strategy = TranslationStrategy.LLM_ASSISTED
        
        # Gather detailed configuration based on the chosen strategy
        strategy_config = self._configure_strategy(combined_strategy, proof_ir, target_prover)
        
        return {
            "strategy": combined_strategy,
            "config": strategy_config,
            "domain_strategy": domain_strategy,
            "pattern_strategy": pattern_strategy,
            "prover_strategy": prover_strategy
        }
    
    def _get_domain_strategy(self, domain: str, target_prover: str) -> TranslationStrategy:
        """
        Get a domain-specific strategy.
        
        Args:
            domain: The mathematical domain
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy
        """
        if domain in self.domain_strategy_mapping:
            strategies = self.domain_strategy_mapping[domain]
            
            # Check for prover-specific strategy
            if target_prover in strategies:
                return strategies[target_prover]
            
            # Default strategy for this domain
            if "default" in strategies:
                return strategies["default"]
        
        # Default to direct strategy
        return TranslationStrategy.DIRECT
    
    def _get_pattern_strategy(self, pattern: str, target_prover: str) -> TranslationStrategy:
        """
        Get a pattern-specific strategy.
        
        Args:
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy
        """
        if pattern in self.pattern_strategy_mapping:
            strategies = self.pattern_strategy_mapping[pattern]
            
            # Check for prover-specific strategy
            if target_prover in strategies:
                return strategies[target_prover]
            
            # Default strategy for this pattern
            if "default" in strategies:
                return strategies["default"]
        
        # Default to direct strategy
        return TranslationStrategy.DIRECT
    
    def _get_prover_strategy(self, target_prover: str) -> TranslationStrategy:
        """
        Get a prover-specific default strategy.
        
        Args:
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy
        """
        # Default strategies per prover
        prover_defaults = {
            "coq": TranslationStrategy.TACTIC_BASED,
            "lean": TranslationStrategy.TACTIC_BASED,
            "isabelle": TranslationStrategy.DECLARATIVE,
            "hol": TranslationStrategy.TERM_BASED
        }
        
        return prover_defaults.get(target_prover.lower(), TranslationStrategy.DIRECT)
    
    def _combine_strategies(self, domain_strategy: TranslationStrategy, 
                           pattern_strategy: TranslationStrategy,
                           prover_strategy: TranslationStrategy,
                           weights: Tuple[float, float, float]) -> TranslationStrategy:
        """
        Combine strategies with weighted voting.
        
        Args:
            domain_strategy: Domain-based strategy
            pattern_strategy: Pattern-based strategy
            prover_strategy: Prover-based strategy
            weights: Weights for each strategy type
            
        Returns:
            The combined strategy
        """
        # Count votes for each strategy
        strategy_votes = {
            domain_strategy: weights[0],
            pattern_strategy: weights[1],
            prover_strategy: weights[2]
        }
        
        # Find strategy with highest vote
        max_votes = 0
        selected_strategy = TranslationStrategy.DIRECT  # Default
        
        for strategy, votes in strategy_votes.items():
            if votes > max_votes:
                max_votes = votes
                selected_strategy = strategy
        
        return selected_strategy
    
    def _initialize_domain_mapping(self) -> Dict[str, Dict[str, TranslationStrategy]]:
        """
        Initialize domain to strategy mappings.
        
        Returns:
            Dictionary mapping domains to strategy recommendations
        """
        return {
            # Number theory
            "11": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Algebra
            "12-20": {
                "default": TranslationStrategy.STRUCTURE_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            },
            
            # Analysis
            "26-42": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TERM_BASED
            },
            
            # Topology
            "54-55": {
                "default": TranslationStrategy.DECLARATIVE,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            }
        }
    
    def _initialize_pattern_mapping(self) -> Dict[str, Dict[str, TranslationStrategy]]:
        """
        Initialize pattern to strategy mappings.
        
        Returns:
            Dictionary mapping proof patterns to strategy recommendations
        """
        return {
            # Induction proofs
            "induction": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Contradiction proofs
            "contradiction": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Case analysis proofs
            "cases": {
                "default": TranslationStrategy.STRUCTURE_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            },
            
            # Direct proofs
            "direct": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED,
                "isabelle": TranslationStrategy.DECLARATIVE
            },
            
            # Existence proofs
            "existence": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            }
        }
    
    def _configure_strategy(self, strategy: TranslationStrategy, 
                           proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Configure the selected strategy with detailed parameters.
        
        Args:
            strategy: The selected translation strategy
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Strategy configuration
        """
        config = {
            "strategy_name": strategy.value,
            "target_prover": target_prover,
            "parameters": {},
            "suggested_tactics": []
        }
        
        # Configure based on strategy type
        if strategy == TranslationStrategy.DIRECT:
            config["parameters"]["step_by_step"] = True
            config["parameters"]["keep_original_structure"] = True
            
        elif strategy == TranslationStrategy.TACTIC_BASED:
            # Extract relevant tactics from proof IR
            config["suggested_tactics"] = [
                {
                    "tactic_type": tactic.tactic_type.value,
                    "arguments": tactic.arguments,
                    "description": tactic.description
                }
                for tactic in proof_ir.tactics
            ]
            
            # Organize by proof phase
            phases = ["setup", "main", "cleanup"]
            config["parameters"]["proof_phases"] = {phase: [] for phase in phases}
            
            # Assign tactics to phases
            for tactic in proof_ir.tactics:
                if tactic.tactic_type == TacticType.INTRO:
                    config["parameters"]["proof_phases"]["setup"].append(tactic.tactic_type.value)
                elif tactic.tactic_type in [TacticType.AUTO, TacticType.SIMPLIFY]:
                    config["parameters"]["proof_phases"]["cleanup"].append(tactic.tactic_type.value)
                else:
                    config["parameters"]["proof_phases"]["main"].append(tactic.tactic_type.value)
        
        elif strategy == TranslationStrategy.STRUCTURE_BASED:
            # Extract proof structure
            config["parameters"]["proof_structure"] = {
                "has_assumptions": any(node.node_type.value == "assumption" for node in proof_ir.proof_tree),
                "has_case_analysis": any(node.node_type.value == "case" for node in proof_ir.proof_tree),
                "has_induction": any(node.node_type.value in ["induction_base", "induction_step"] 
                                    for node in proof_ir.proof_tree),
                "has_contradiction": any(node.node_type.value == "contradiction" for node in proof_ir.proof_tree)
            }
            
            # Identify key variables
            variables = proof_ir.metadata.get("variables", [])
            config["parameters"]["key_variables"] = variables
        
        elif strategy == TranslationStrategy.DECLARATIVE:
            # For declarative proofs (like Isar in Isabelle)
            config["parameters"]["use_declarative_style"] = True
            config["parameters"]["explicit_justifications"] = True
            
        elif strategy == TranslationStrategy.TERM_BASED:
            # For term-based proofs (direct proof terms)
            config["parameters"]["use_proof_terms"] = True
            config["parameters"]["lambda_notation"] = True
            
        elif strategy == TranslationStrategy.LLM_ASSISTED:
            # For LLM-assisted translation
            config["parameters"]["use_llm"] = True
            config["parameters"]["temperature"] = 0.3
            config["parameters"]["max_tokens"] = 1000
            config["parameters"]["fallback"] = self.fallback_to_llm
        
        return config


# Standalone functions for use in other modules

def select_translation_strategy(proof_ir: ProofIR, target_prover: str, 
                               use_llm: bool = False) -> Dict[str, Any]:
    """
    Select an appropriate translation strategy for a proof.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        use_llm: Whether to use language model assistance
        
    Returns:
        Dictionary with strategy information
    """
    selector = StrategySelector(use_llm=use_llm)
    return selector.select_strategy(proof_ir, target_prover)

def get_optimal_strategy(theorem_text: str, proof_text: str, 
                        target_prover: str) -> Dict[str, Any]:
    """
    Get the optimal translation strategy directly from theorem and proof text.
    
    Args:
        theorem_text: The theorem statement
        proof_text: The proof text
        target_prover: The target theorem prover
        
    Returns:
        Dictionary with strategy information
    """
    # Parse the proof
    parsed_statements, proof_structure = parse_proof(proof_text)
    
    # Detect domain
    domain_info = detect_domain(theorem_text, proof_text)
    
    # Recognize pattern
    pattern_info = recognize_pattern(proof_text)
    
    # Create a minimal proof IR
    from ir.proof_builder import build_proof_ir
    proof_ir = build_proof_ir(
        parsed_statements=parsed_statements,
        proof_structure=proof_structure,
        original_theorem=theorem_text,
        original_proof=proof_text
    )
    
    # Select strategy
    selector = StrategySelector()
    return selector.select_strategy(proof_ir, target_prover)