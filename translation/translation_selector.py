"""
Translation strategy selector for proof translation.
Selects domain-specific translation strategies based on proof characteristics.
"""

from enum import Enum
import re
import logging
from typing import Dict, List, Optional, Any, Union

from ir.proof_ir import ProofIR, TacticType, TacticInfo
from knowledge_base.domain_kb import DomainKnowledgeBase

# Configure logging
logger = logging.getLogger("translation_selector")

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
    
    # Domain-specific strategies
    NUMBER_THEORY = "number_theory"
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"

class TranslationSelector:
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
        
        # Initialize the knowledge base
        try:
            self.kb = DomainKnowledgeBase()
            logger.info("Initialized knowledge base for strategy selection")
        except Exception as e:
            logger.warning(f"Error initializing knowledge base: {e}. Using limited strategies.")
            self.kb = None
        
        # Initialize domain-specific strategy generators
        self.domain_strategies = self._initialize_domain_strategies()
        
        # Initialize pattern-specific strategy generators
        self.pattern_strategies = self._initialize_pattern_strategies()
    
    def _initialize_domain_strategies(self) -> Dict[str, Dict[str, TranslationStrategy]]:
        """
        Initialize domain-specific strategy mappings.
        
        Returns:
            Dictionary mapping domains to strategy recommendations
        """
        return {
            # Number theory
            "11": {
                "default": TranslationStrategy.NUMBER_THEORY,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Algebra domains
            "12": {"default": TranslationStrategy.ALGEBRA},
            "13": {"default": TranslationStrategy.ALGEBRA},
            "15": {"default": TranslationStrategy.ALGEBRA},
            "16": {"default": TranslationStrategy.ALGEBRA},
            "17": {"default": TranslationStrategy.ALGEBRA},
            "18": {"default": TranslationStrategy.ALGEBRA},
            "19": {"default": TranslationStrategy.ALGEBRA},
            "20": {"default": TranslationStrategy.ALGEBRA},
            
            # Analysis domains
            "26": {"default": TranslationStrategy.ANALYSIS},
            "28": {"default": TranslationStrategy.ANALYSIS},
            "30": {"default": TranslationStrategy.ANALYSIS},
            "31": {"default": TranslationStrategy.ANALYSIS},
            "32": {"default": TranslationStrategy.ANALYSIS},
            "33": {"default": TranslationStrategy.ANALYSIS},
            "34": {"default": TranslationStrategy.ANALYSIS},
            "35": {"default": TranslationStrategy.ANALYSIS},
            "40": {"default": TranslationStrategy.ANALYSIS},
            "41": {"default": TranslationStrategy.ANALYSIS},
            "42": {"default": TranslationStrategy.ANALYSIS},
            
            # Topology domains
            "54": {"default": TranslationStrategy.TOPOLOGY},
            "55": {"default": TranslationStrategy.TOPOLOGY}
        }
    
    def _initialize_pattern_strategies(self) -> Dict[str, Dict[str, TranslationStrategy]]:
        """
        Initialize pattern-specific strategy mappings.
        
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
            
            # Mathematical induction (special case of induction)
            "mathematical_induction": {
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
            "case_analysis": {
                "default": TranslationStrategy.STRUCTURE_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            },
            
            # Cases (variant of case analysis)
            "cases": {
                "default": TranslationStrategy.STRUCTURE_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            },
            
            # Direct proofs
            "direct": {
                "default": TranslationStrategy.DIRECT,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Existence proofs
            "existence": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            },
            
            # Uniqueness proofs
            "uniqueness": {
                "default": TranslationStrategy.STRUCTURE_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.STRUCTURE_BASED
            },
            
            # Evenness proofs
            "evenness_proof": {
                "default": TranslationStrategy.TACTIC_BASED,
                "coq": TranslationStrategy.TACTIC_BASED,
                "lean": TranslationStrategy.TACTIC_BASED
            }
        }
    
    def select_strategy(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Select an appropriate translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        logger.info(f"Selecting strategy for {target_prover} translation")
        
        # Extract domain and pattern information
        domain = proof_ir.domain.get("primary_domain", "")
        pattern = proof_ir.pattern.get("name", "")
        
        logger.info(f"Proof domain: {domain}, pattern: {pattern}")
        
        # Force LLM-assisted strategy if requested
        if self.use_llm:
            return self._create_llm_strategy_info(domain, pattern, target_prover)
        
        # Get domain-specific strategy
        domain_strategy = self._get_domain_strategy(domain, target_prover)
        
        # Get pattern-specific strategy
        pattern_strategy = self._get_pattern_strategy(pattern, target_prover)
        
        # Get target prover default strategy
        prover_strategy = self._get_prover_strategy(target_prover)
        
        # Combine strategies with appropriate weights
        combined_strategy = self._combine_strategies(
            domain_strategy, pattern_strategy, prover_strategy,
            weights=(0.3, 0.5, 0.2)  # Pattern strategy has higher weight
        )
        
        logger.info(f"Selected strategy: {combined_strategy.value}")
        
        # Build strategy configuration
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
        # Check for exact domain match
        if domain in self.domain_strategies:
            strategies = self.domain_strategies[domain]
            
            # Check for prover-specific strategy
            if target_prover in strategies:
                return strategies[target_prover]
            
            # Default strategy for this domain
            if "default" in strategies:
                return strategies["default"]
        
        # Check for domain range match (e.g., for "12-20" Algebra domains)
        for domain_range, strategies in self.domain_strategies.items():
            if "-" in domain_range:
                start, end = domain_range.split("-")
                if start <= domain <= end:
                    # Check for prover-specific strategy
                    if target_prover in strategies:
                        return strategies[target_prover]
                    
                    # Default strategy for this domain range
                    if "default" in strategies:
                        return strategies["default"]
        
        # Use knowledge base if available
        if self.kb:
            kb_strategy = self._get_kb_domain_strategy(domain, target_prover)
            if kb_strategy:
                return kb_strategy
        
        # Default to DIRECT strategy if no match found
        return TranslationStrategy.DIRECT
    
    def _get_kb_domain_strategy(self, domain: str, target_prover: str) -> Optional[TranslationStrategy]:
        """
        Get a domain strategy from the knowledge base.
        
        Args:
            domain: The mathematical domain
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy or None if not found
        """
        if not self.kb:
            return None
        
        # Try to get domain info from the knowledge base
        domain_info = self.kb.get_domain_info(domain)
        if not domain_info:
            return None
        
        # Check if there's translation strategy information
        if "translation_strategy" in domain_info:
            strategy_name = domain_info["translation_strategy"].get(target_prover) or domain_info["translation_strategy"].get("default")
            if strategy_name:
                try:
                    return TranslationStrategy(strategy_name)
                except ValueError:
                    logger.warning(f"Unknown strategy name in knowledge base: {strategy_name}")
        
        return None
    
    def _get_pattern_strategy(self, pattern: str, target_prover: str) -> TranslationStrategy:
        """
        Get a pattern-specific strategy.
        
        Args:
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy
        """
        if not pattern:
            return TranslationStrategy.DIRECT
        
        if pattern in self.pattern_strategies:
            strategies = self.pattern_strategies[pattern]
            
            # Check for prover-specific strategy
            if target_prover in strategies:
                return strategies[target_prover]
            
            # Default strategy for this pattern
            if "default" in strategies:
                return strategies["default"]
        
        # Use knowledge base if available
        if self.kb:
            kb_strategy = self._get_kb_pattern_strategy(pattern, target_prover)
            if kb_strategy:
                return kb_strategy
        
        # Default to TACTIC_BASED strategy for most patterns
        return TranslationStrategy.TACTIC_BASED
    
    def _get_kb_pattern_strategy(self, pattern: str, target_prover: str) -> Optional[TranslationStrategy]:
        """
        Get a pattern strategy from the knowledge base.
        
        Args:
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            A translation strategy or None if not found
        """
        if not self.kb or not hasattr(self.kb, 'patterns'):
            return None
        
        # Try to get pattern info from the knowledge base
        if pattern in self.kb.patterns:
            pattern_info = self.kb.patterns[pattern]
            
            # Check if there's translation strategy information
            if "translation_strategy" in pattern_info:
                strategy_name = (pattern_info["translation_strategy"].get(target_prover) or 
                                pattern_info["translation_strategy"].get("default"))
                if strategy_name:
                    try:
                        return TranslationStrategy(strategy_name)
                    except ValueError:
                        logger.warning(f"Unknown strategy name in knowledge base: {strategy_name}")
        
        return None
    
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
    
    def _combine_strategies(self, 
                           domain_strategy: TranslationStrategy, 
                           pattern_strategy: TranslationStrategy,
                           prover_strategy: TranslationStrategy,
                           weights: tuple) -> TranslationStrategy:
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
        strategy_votes = {}
        strategy_votes[domain_strategy] = weights[0]
        strategy_votes[pattern_strategy] = weights[1]
        strategy_votes[prover_strategy] = weights[2]
        
        # Special case: if domain strategy is domain-specific, give it higher weight
        if domain_strategy in [
            TranslationStrategy.NUMBER_THEORY,
            TranslationStrategy.ALGEBRA,
            TranslationStrategy.ANALYSIS,
            TranslationStrategy.TOPOLOGY
        ]:
            strategy_votes[domain_strategy] += 0.2
        
        # Find strategy with highest vote
        max_votes = 0
        selected_strategy = TranslationStrategy.DIRECT  # Default
        
        for strategy, votes in strategy_votes.items():
            if votes > max_votes:
                max_votes = votes
                selected_strategy = strategy
        
        # Special case handling for domain-specific strategy refinement
        if selected_strategy == TranslationStrategy.NUMBER_THEORY:
            # For number theory, we prefer tactic-based approach for most provers
            return TranslationStrategy.TACTIC_BASED
        elif selected_strategy == TranslationStrategy.ALGEBRA:
            # For algebra, we prefer structure-based approach for most provers
            return TranslationStrategy.STRUCTURE_BASED
        elif selected_strategy == TranslationStrategy.ANALYSIS:
            # For analysis, we prefer tactic-based approach for most provers
            return TranslationStrategy.TACTIC_BASED
        elif selected_strategy == TranslationStrategy.TOPOLOGY:
            # For topology, we prefer declarative approach for most provers
            return TranslationStrategy.DECLARATIVE
        
        return selected_strategy
    
    def _create_llm_strategy_info(self, domain: str, pattern: str, target_prover: str) -> Dict[str, Any]:
        """
        Create strategy information for LLM-assisted translation.
        
        Args:
            domain: The mathematical domain
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Look up domain information for context
        domain_info = {}
        if self.kb:
            domain_info = self.kb.get_domain_info(domain)
        
        # Look up pattern information for context
        pattern_info = {}
        if self.kb and hasattr(self.kb, 'patterns') and pattern in self.kb.patterns:
            pattern_info = self.kb.patterns[pattern]
        
        # Build LLM strategy config
        config = {
            "strategy_name": TranslationStrategy.LLM_ASSISTED.value,
            "target_prover": target_prover,
            "parameters": {
                "use_llm": True,
                "temperature": 0.3,
                "max_tokens": 1000,
                "fallback": self.fallback_to_llm,
                "domain_context": {
                    "domain": domain,
                    "domain_name": domain_info.get("name", ""),
                    "pattern": pattern
                }
            }
        }
        
        # Add examples if available
        if self.kb:
            examples = self.kb.get_examples(domain, pattern, target_prover)
            if examples:
                config["parameters"]["examples"] = examples
        
        return {
            "strategy": TranslationStrategy.LLM_ASSISTED,
            "config": config,
            "domain_strategy": TranslationStrategy.LLM_ASSISTED,
            "pattern_strategy": TranslationStrategy.LLM_ASSISTED,
            "prover_strategy": TranslationStrategy.LLM_ASSISTED
        }
    
    def _configure_strategy(self, 
                           strategy: TranslationStrategy, 
                           proof_ir: ProofIR, 
                           target_prover: str) -> Dict[str, Any]:
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
            
            # Add library dependencies based on domain context
            if proof_ir.domain_context and proof_ir.domain_context.libraries:
                config["parameters"]["libraries"] = proof_ir.domain_context.libraries
        
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
        
        # Add domain-specific configuration
        domain = proof_ir.domain.get("primary_domain", "")
        domain_config = self._get_domain_specific_config(domain, strategy, target_prover)
        if domain_config:
            # Merge domain-specific parameters
            config["parameters"].update(domain_config.get("parameters", {}))
            
            # Add domain-specific tactics
            domain_tactics = domain_config.get("suggested_tactics", [])
            if domain_tactics:
                config["suggested_tactics"].extend(domain_tactics)
        
        # Add pattern-specific configuration
        pattern = proof_ir.pattern.get("name", "")
        pattern_config = self._get_pattern_specific_config(pattern, strategy, target_prover)
        if pattern_config:
            # Merge pattern-specific parameters
            config["parameters"].update(pattern_config.get("parameters", {}))
            
            # Add pattern-specific tactics
            pattern_tactics = pattern_config.get("suggested_tactics", [])
            if pattern_tactics:
                config["suggested_tactics"].extend(pattern_tactics)
        
        return config
    
    def _get_domain_specific_config(self, 
                                   domain: str, 
                                   strategy: TranslationStrategy, 
                                   target_prover: str) -> Optional[Dict[str, Any]]:
        """
        Get domain-specific configuration.
        
        Args:
            domain: The mathematical domain
            strategy: The selected strategy
            target_prover: The target theorem prover
            
        Returns:
            Domain-specific configuration or None
        """
        if not domain:
            return None
        
        # Number Theory configuration
        if domain == "11":
            return {
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": ["Arith", "ZArith", "Lia"] if target_prover == "coq" else 
                                       ["Mathlib.Data.Nat.Basic", "Mathlib.Tactic.NormNum"]
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "custom",
                        "arguments": ["lia"] if target_prover == "coq" else ["norm_num"],
                        "description": "Use arithmetic solver"
                    },
                    {
                        "tactic_type": "simplify",
                        "arguments": [],
                        "description": "Simplify expressions"
                    }
                ]
            }
        
        # Algebra configuration (domains 12-20)
        elif domain in ["12", "13", "15", "16", "17", "18", "19", "20"]:
            return {
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": ["Ring", "Field"] if target_prover == "coq" else 
                                       ["Mathlib.Algebra.Ring.Basic", "Mathlib.Tactic.Ring"]
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "custom",
                        "arguments": ["ring"] if target_prover == "coq" else ["ring"],
                        "description": "Use ring solver"
                    },
                    {
                        "tactic_type": "simplify",
                        "arguments": [],
                        "description": "Simplify expressions"
                    }
                ]
            }
        
        # Analysis configuration (domains 26-42)
        elif domain in ["26", "28", "30", "31", "32", "33", "34", "35", "40", "41", "42"]:
            return {
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": ["Reals"] if target_prover == "coq" else 
                                       ["Mathlib.Analysis.RealFunction"]
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "custom",
                        "arguments": ["field"] if target_prover == "coq" else ["field_simp"],
                        "description": "Use field solver"
                    }
                ]
            }
        
        # Topology configuration (domains 54-55)
        elif domain in ["54", "55"]:
            return {
                "parameters": {
                    "use_automation": False,
                    "prefer_libraries": ["Topology"] if target_prover == "coq" else 
                                       ["Mathlib.Topology.Basic"]
                },
                "suggested_tactics": []
            }
        
        # Try to get domain config from knowledge base
        if self.kb:
            return self._get_kb_domain_config(domain, strategy, target_prover)
        
        return None
    
    def _get_kb_domain_config(self, 
                             domain: str, 
                             strategy: TranslationStrategy, 
                             target_prover: str) -> Optional[Dict[str, Any]]:
        """
        Get domain configuration from the knowledge base.
        
        Args:
            domain: The mathematical domain
            strategy: The selected strategy
            target_prover: The target theorem prover
            
        Returns:
            Domain configuration or None
        """
        if not self.kb:
            return None
        
        # Get domain info
        domain_info = self.kb.get_domain_info(domain)
        if not domain_info:
            return None
        
        # Check if there's translation configuration
        if "translation_config" in domain_info:
            config = domain_info["translation_config"].get(target_prover) or domain_info["translation_config"].get("default")
            if config:
                return config
        
        # Build configuration from available information
        config = {"parameters": {}, "suggested_tactics": []}
        
        # Add libraries
        libraries = self.kb.get_domain_libraries(domain, target_prover)
        if libraries:
            config["parameters"]["prefer_libraries"] = libraries
        
        # Add tactics
        tactics = self.kb.get_domain_tactics(domain, target_prover)
        if tactics:
            config["suggested_tactics"] = tactics
        
        return config if (config["parameters"] or config["suggested_tactics"]) else None


# Standalone domain-specific strategy generators

class NumberTheoryStrategy:
    """
    Translation strategy for number theory proofs.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the number theory strategy.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply number theory-specific translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Get pattern from proof IR
        pattern = proof_ir.pattern.get("name", "")
        
        # Get required libraries for number theory
        libraries = []
        if self.kb:
            libraries = self.kb.get_domain_libraries("11", target_prover)
        else:
            # Default libraries if knowledge base not available
            if target_prover == "coq":
                libraries = ["Arith", "ZArith", "Lia"]
            else:  # lean
                libraries = ["Mathlib.Data.Nat.Basic", "Mathlib.Tactic.NormNum"]
        
        # Get tactics based on pattern
        tactics = []
        if pattern == "induction":
            tactics = self._enhance_induction_tactics(proof_ir, target_prover)
        elif pattern == "contradiction":
            tactics = self._enhance_contradiction_tactics(proof_ir, target_prover)
        elif pattern == "evenness_proof":
            tactics = self._enhance_evenness_tactics(proof_ir, target_prover)
        else:
            # General number theory tactics
            tactics = self._get_general_tactics(proof_ir, target_prover)
        
        return {
            "strategy": TranslationStrategy.NUMBER_THEORY,
            "config": {
                "strategy_name": TranslationStrategy.NUMBER_THEORY.value,
                "target_prover": target_prover,
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": libraries
                },
                "suggested_tactics": tactics
            }
        }
    
    def _enhance_induction_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance induction tactics for number theory.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            List of enhanced tactics
        """
        tactics = []
        
        # Find the variable being inducted on from tactics or metadata
        induction_var = None
        for tactic in proof_ir.tactics:
            if tactic.tactic_type == TacticType.INDUCTION and tactic.arguments:
                induction_var = tactic.arguments[0]
                break
                
        if not induction_var:
            # Try to get from variables list
            variables = proof_ir.metadata.get("variables", [])
            induction_var = variables[0] if variables else "n"
        
        # Add induction tactic
        tactics.append({
            "tactic_type": "induction",
            "arguments": [induction_var],
            "description": f"Apply induction on {induction_var}"
        })
        
        # Add number theory-specific tactics
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["lia"],
                "description": "Use linear integer arithmetic"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["norm_num"],
                "description": "Normalize numeric expressions"
            })
        
        return tactics
    
    def _enhance_contradiction_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance contradiction tactics for number theory.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            List of enhanced tactics
        """
        tactics = []
        
        # Add contradiction setup
        tactics.append({
            "tactic_type": "contradiction",
            "arguments": [],
            "description": "Set up proof by contradiction"
        })
        
        # Add number theory-specific tactics
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["nia"],
                "description": "Use non-linear integer arithmetic"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["linarith"],
                "description": "Use linear arithmetic reasoning"
            })
        
        return tactics
    
    def _enhance_evenness_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Enhance evenness proof tactics for number theory.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            List of enhanced tactics
        """
        tactics = []
        
        # Add tactics for evenness proofs
        
        # First, analyze proof tree to find the expression being proven even
        expression = None
        for node in proof_ir.proof_tree:
            # Look for nodes with "even" in their content
            if hasattr(node, "content") and "even" in str(node.content).lower():
                content = str(node.content)
                # Try to extract the expression
                match = re.search(r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+even', content, re.IGNORECASE)
                if match:
                    expression = match.group(1)
                    break
                
                # Alternative pattern with "divisible by 2"
                match = re.search(r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+divisible\s+by\s+2', content, re.IGNORECASE)
                if match:
                    expression = match.group(1)
                    break
                
                # Alternative pattern with "multiple of 2"
                match = re.search(r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+a\s+multiple\s+of\s+2', content, re.IGNORECASE)
                if match:
                    expression = match.group(1)
                    break
                
                # Simple cases like "x+x"
                match = re.search(r'([a-zA-Z])\s*\+\s*\1', content)
                if match:
                    expression = f"{match.group(1)}+{match.group(1)}"
                    break
        
        # Add exists tactic for providing a witness
        if expression:
            # If the expression is of the form x+x, the witness is x
            if re.match(r'([a-zA-Z])\s*\+\s*\1', expression):
                var = re.match(r'([a-zA-Z])\s*\+\s*\1', expression).group(1)
                tactics.append({
                    "tactic_type": "exists",
                    "arguments": [var],
                    "description": f"Provide witness {var} for evenness of {expression}"
                })
            else:
                # Generic witness
                tactics.append({
                    "tactic_type": "exists",
                    "arguments": ["k"],
                    "description": f"Provide witness k for evenness of {expression}"
                })
        else:
            # Generic exists tactic
            tactics.append({
                "tactic_type": "exists",
                "arguments": [],
                "description": "Provide witness for evenness"
            })
        
        # Add ring tactic for algebraic simplification
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["ring"],
                "description": "Use ring for algebraic equality"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["ring"],
                "description": "Use ring for algebraic equality"
            })
        
        return tactics
    
    def _get_general_tactics(self, proof_ir: ProofIR, target_prover: str) -> List[Dict[str, Any]]:
        """
        Get general number theory tactics.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Add intros tactic for variables
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            tactics.append({
                "tactic_type": "intro",
                "arguments": variables,
                "description": f"Introduce variables: {', '.join(variables)}"
            })
        
        # Add simplification tactic
        tactics.append({
            "tactic_type": "simplify",
            "arguments": [],
            "description": "Simplify expressions"
        })
        
        # Add number theory automation
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["lia"],
                "description": "Use linear integer arithmetic"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["norm_num"],
                "description": "Normalize numeric expressions"
            })
        
        return tactics


class AlgebraStrategy:
    """
    Translation strategy for algebraic proofs.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the algebra strategy.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply algebra-specific translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Get pattern from proof IR
        pattern = proof_ir.pattern.get("name", "")
        
        # Get required libraries for algebra
        libraries = []
        if self.kb:
            libraries = self.kb.get_domain_libraries("12-20", target_prover)
        else:
            # Default libraries if knowledge base not available
            if target_prover == "coq":
                libraries = ["Ring", "Field", "Algebra"]
            else:  # lean
                libraries = ["Mathlib.Algebra.Ring.Basic", "Mathlib.Tactic.Ring"]
        
        # Get tactics based on pattern
        tactics = self._get_algebra_tactics(proof_ir, pattern, target_prover)
        
        return {
            "strategy": TranslationStrategy.ALGEBRA,
            "config": {
                "strategy_name": TranslationStrategy.ALGEBRA.value,
                "target_prover": target_prover,
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": libraries,
                    "structure_based": True
                },
                "suggested_tactics": tactics
            }
        }
    
    def _get_algebra_tactics(self, proof_ir: ProofIR, pattern: str, target_prover: str) -> List[Dict[str, Any]]:
        """
        Get algebra-specific tactics.
        
        Args:
            proof_ir: The proof intermediate representation
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Add intros tactic for variables
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            tactics.append({
                "tactic_type": "intro",
                "arguments": variables,
                "description": f"Introduce variables: {', '.join(variables)}"
            })
        
        # Add algebra-specific tactics based on pattern
        if pattern == "induction":
            # Not common in pure algebra, but can occur
            tactics.append({
                "tactic_type": "induction",
                "arguments": [],
                "description": "Apply induction"
            })
        
        # Add ring tactic for algebraic simplification
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["ring"],
                "description": "Use ring for algebraic equalities"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["ring"],
                "description": "Use ring for algebraic equalities"
            })
        
        # Add field tactic for field equalities
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["field"],
                "description": "Use field for rational expressions"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["field_simp"],
                "description": "Simplify field expressions"
            })
        
        return tactics


class AnalysisStrategy:
    """
    Translation strategy for analysis proofs.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the analysis strategy.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply analysis-specific translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Get pattern from proof IR
        pattern = proof_ir.pattern.get("name", "")
        
        # Get required libraries for analysis
        libraries = []
        if self.kb:
            libraries = self.kb.get_domain_libraries("26-42", target_prover)
        else:
            # Default libraries if knowledge base not available
            if target_prover == "coq":
                libraries = ["Reals", "RiemannInt", "Fourier"]
            else:  # lean
                libraries = ["Mathlib.Analysis.RealFunction", "Mathlib.Analysis.Calculus.Deriv.Basic"]
        
        # Get tactics based on pattern
        tactics = self._get_analysis_tactics(proof_ir, pattern, target_prover)
        
        return {
            "strategy": TranslationStrategy.ANALYSIS,
            "config": {
                "strategy_name": TranslationStrategy.ANALYSIS.value,
                "target_prover": target_prover,
                "parameters": {
                    "use_automation": True,
                    "prefer_libraries": libraries,
                    "epsilon_delta": "analysis" in pattern.lower()
                },
                "suggested_tactics": tactics
            }
        }
    
    def _get_analysis_tactics(self, proof_ir: ProofIR, pattern: str, target_prover: str) -> List[Dict[str, Any]]:
        """
        Get analysis-specific tactics.
        
        Args:
            proof_ir: The proof intermediate representation
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Add intros tactic for variables
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            tactics.append({
                "tactic_type": "intro",
                "arguments": variables,
                "description": f"Introduce variables: {', '.join(variables)}"
            })
        
        # Add analysis-specific tactics
        if target_prover == "coq":
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["field"],
                "description": "Use field for real-valued expressions"
            })
        else:  # lean
            tactics.append({
                "tactic_type": "custom",
                "arguments": ["field_simp"],
                "description": "Simplify field expressions"
            })
        
        # Add simplification tactic
        tactics.append({
            "tactic_type": "simplify",
            "arguments": [],
            "description": "Simplify expressions"
        })
        
        return tactics


class TopologyStrategy:
    """
    Translation strategy for topology proofs.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the topology strategy.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
    
    def apply(self, proof_ir: ProofIR, target_prover: str) -> Dict[str, Any]:
        """
        Apply topology-specific translation strategy.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with strategy information
        """
        # Get pattern from proof IR
        pattern = proof_ir.pattern.get("name", "")
        
        # Get required libraries for topology
        libraries = []
        if self.kb:
            libraries = self.kb.get_domain_libraries("54-55", target_prover)
        else:
            # Default libraries if knowledge base not available
            if target_prover == "coq":
                libraries = ["Topology"]
            else:  # lean
                libraries = ["Mathlib.Topology.Basic", "Mathlib.Topology.MetricSpace.Basic"]
        
        # Get tactics based on pattern
        tactics = self._get_topology_tactics(proof_ir, pattern, target_prover)
        
        return {
            "strategy": TranslationStrategy.TOPOLOGY,
            "config": {
                "strategy_name": TranslationStrategy.TOPOLOGY.value,
                "target_prover": target_prover,
                "parameters": {
                    "use_automation": False,
                    "prefer_libraries": libraries,
                    "declarative_style": True
                },
                "suggested_tactics": tactics
            }
        }
    
    def _get_topology_tactics(self, proof_ir: ProofIR, pattern: str, target_prover: str) -> List[Dict[str, Any]]:
        """
        Get topology-specific tactics.
        
        Args:
            proof_ir: The proof intermediate representation
            pattern: The proof pattern
            target_prover: The target theorem prover
            
        Returns:
            List of tactics
        """
        tactics = []
        
        # Add intros tactic for variables
        variables = proof_ir.metadata.get("variables", [])
        if variables:
            tactics.append({
                "tactic_type": "intro",
                "arguments": variables,
                "description": f"Introduce variables: {', '.join(variables)}"
            })
        
        # Add simplification tactic
        tactics.append({
            "tactic_type": "simplify",
            "arguments": [],
            "description": "Simplify expressions"
        })
        
        return tactics


# Standalone utility functions

def select_translation_strategy(proof_ir: ProofIR, target_prover: str, use_llm: bool = False) -> Dict[str, Any]:
    """
    Select an appropriate translation strategy for a proof.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        use_llm: Whether to use language model assistance
        
    Returns:
        Dictionary with strategy information
    """
    selector = TranslationSelector(use_llm=use_llm)
    return selector.select_strategy(proof_ir, target_prover)

def get_domain_strategies() -> Dict[str, str]:
    """
    Get available domain-specific strategies.
    
    Returns:
        Dictionary mapping domain codes to strategy names
    """
    return {
        "11": "NUMBER_THEORY",
        "12-20": "ALGEBRA",
        "26-42": "ANALYSIS",
        "54-55": "TOPOLOGY"
    }

def get_pattern_strategies() -> Dict[str, str]:
    """
    Get available pattern-specific strategies.
    
    Returns:
        Dictionary mapping pattern names to strategy names
    """
    return {
        "induction": "TACTIC_BASED",
        "contradiction": "TACTIC_BASED",
        "case_analysis": "STRUCTURE_BASED",
        "direct": "DIRECT",
        "evenness_proof": "TACTIC_BASED"
    }

def create_strategy_for_domain(domain: str, kb=None) -> Any:
    """
    Create a domain-specific strategy instance.
    
    Args:
        domain: The domain code
        kb: Optional knowledge base
        
    Returns:
        A strategy instance
    """
    if domain == "11":
        return NumberTheoryStrategy(kb)
    elif domain in ["12", "13", "15", "16", "17", "18", "19", "20"] or domain == "12-20":
        return AlgebraStrategy(kb)
    elif domain in ["26", "28", "30", "31", "32", "33", "34", "35", "40", "41", "42"] or domain == "26-42":
        return AnalysisStrategy(kb)
    elif domain in ["54", "55"] or domain == "54-55":
        return TopologyStrategy(kb)
    else:
        return None
    
    def _get_pattern_specific_config(self, 
                                    pattern: str, 
                                    strategy: TranslationStrategy, 
                                    target_prover: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern-specific configuration.
        
        Args:
            pattern: The proof pattern
            strategy: The selected strategy
            target_prover: The target theorem prover
            
        Returns:
            Pattern-specific configuration or None
        """
        if not pattern:
            return None
        
        # Induction pattern
        if pattern in ["induction", "mathematical_induction"]:
            return {
                "parameters": {
                    "induction_approach": True,
                    "generate_base_case": True,
                    "generate_inductive_step": True
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "induction",
                        "arguments": [],
                        "description": "Apply induction"
                    }
                ]
            }
        
        # Contradiction pattern
        elif pattern == "contradiction":
            return {
                "parameters": {
                    "contradiction_approach": True,
                    "apply_negation": True
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "contradiction",
                        "arguments": [],
                        "description": "Apply contradiction"
                    }
                ]
            }
        
        # Case analysis pattern
        elif pattern in ["case_analysis", "cases"]:
            return {
                "parameters": {
                    "case_approach": True,
                    "generate_cases": True
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "case_analysis",
                        "arguments": [],
                        "description": "Apply case analysis"
                    }
                ]
            }
        
        # Direct pattern
        elif pattern == "direct":
            return {
                "parameters": {
                    "step_by_step": True
                },
                "suggested_tactics": []
            }
        
        # Evenness proof pattern
        elif pattern == "evenness_proof":
            return {
                "parameters": {
                    "evenness_approach": True,
                    "generate_witness": True
                },
                "suggested_tactics": [
                    {
                        "tactic_type": "exists",
                        "arguments": [],
                        "description": "Provide witness for evenness"
                    },
                    {
                        "tactic_type": "custom",
                        "arguments": ["ring"],
                        "description": "Use ring solver"
                    }
                ]
            }
        
        # Try to get pattern config from knowledge base
        if self.kb:
            return self._get_kb_pattern_config(pattern, strategy, target_prover)
        
        return None
    
    def _get_kb_pattern_config(self, 
                              pattern: str, 
                              strategy: TranslationStrategy, 
                              target_prover: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern configuration from the knowledge base.
        
        Args:
            pattern: The proof pattern
            strategy: The selected strategy
            target_prover: The target theorem prover
            
        Returns:
            Pattern configuration or None
        """
        if not self.kb or not hasattr(self.kb, 'patterns'):
            return None
        
        # Get pattern info
        if pattern not in self.kb.patterns:
            return None
            
        pattern_info = self.kb.patterns[pattern]
        
        # Check if there's translation configuration
        if "translation_config" in pattern_info:
            config = pattern_info["translation_config"].get(target_prover) or pattern_info["translation_config"].get("default")
            if config:
                return config
        
        # Build configuration from available information
        config = {"parameters": {}, "suggested_tactics": []}
        
        # Add tactics from pattern info
        if "tactics" in pattern_info and target_prover in pattern_info["tactics"]:
            config["suggested_tactics"] = pattern_info["tactics"][target_prover]
        
        return config if (config["parameters"] or config["suggested_tactics"]) else None