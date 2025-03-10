"""
Lean mapper for mathematical concepts.
Maps mathematical concepts to their Lean representation.
"""

from typing import Dict, List, Any, Optional, Union
import re

class LeanMapper:
    """
    Maps mathematical concepts to their Lean representations.
    """
    
    def __init__(self):
        """Initialize the Lean mapper."""
        self.concept_mappings = self._initialize_concept_mappings()
        self.library_mappings = self._initialize_library_mappings()
        self.notation_mappings = self._initialize_notation_mappings()
        self.tactic_mappings = self._initialize_tactic_mappings()
    
    def map_concept(self, concept: str, domain: Optional[str] = None) -> str:
        """
        Map a mathematical concept to its Lean representation.
        
        Args:
            concept: The concept to map
            domain: Optional domain for context
            
        Returns:
            The Lean representation
        """
        # Check for exact match
        if concept in self.concept_mappings:
            return self.concept_mappings[concept]
        
        # Check for domain-specific mappings
        if domain:
            domain_mappings = self._get_domain_mappings(domain)
            if concept in domain_mappings:
                return domain_mappings[concept]
        
        # No mapping found, return as-is
        return concept
    
    def get_library_imports(self, concepts: List[str], domain: Optional[str] = None) -> List[str]:
        """
        Get required library imports for a set of concepts.
        
        Args:
            concepts: List of concepts
            domain: Optional domain for context
            
        Returns:
            List of required library imports
        """
        required_imports = set()
        
        # Add imports for each concept
        for concept in concepts:
            # Check if the concept requires specific imports
            for library, library_concepts in self.library_mappings.items():
                if concept in library_concepts:
                    required_imports.add(library)
        
        # Add domain-specific imports
        if domain:
            domain_imports = self._get_domain_imports(domain)
            required_imports.update(domain_imports)
        
        return sorted(list(required_imports))
    
    def map_notation(self, expression: str) -> str:
        """
        Map mathematical notation to Lean notation.
        
        Args:
            expression: The mathematical expression
            
        Returns:
            The Lean notation
        """
        result = expression
        
        # Apply notation mappings
        for pattern, replacement in self.notation_mappings.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def map_tactic(self, tactic_name: str, args: Optional[List[str]] = None) -> str:
        """
        Map a tactic name to Lean tactic syntax.
        
        Args:
            tactic_name: The tactic name
            args: Optional arguments for the tactic
            
        Returns:
            The Lean tactic syntax
        """
        if tactic_name in self.tactic_mappings:
            tactic_template = self.tactic_mappings[tactic_name]
            
            if args:
                # Insert arguments into the template
                if "{args}" in tactic_template:
                    args_str = " ".join(args)
                    return tactic_template.replace("{args}", args_str)
                else:
                    # Append arguments if no placeholder
                    return f"{tactic_template} {' '.join(args)}"
            else:
                # Return template without args
                return tactic_template.replace(" {args}", "").replace("{args}", "")
        
        # No mapping found, return as-is
        return tactic_name
    
    def _initialize_concept_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from mathematical concepts to Lean representations.
        
        Returns:
            Dictionary mapping concepts to Lean representations
        """
        return {
            # Basic number types
            "natural number": "Nat",
            "integer": "Int",
            "rational number": "Rat",
            "real number": "Real",
            "complex number": "Complex",
            
            # Mathematical structures
            "set": "Set",
            "group": "Group",
            "ring": "Ring",
            "field": "Field",
            "vector space": "VectorSpace",
            
            # Logical operators
            "and": "∧",
            "or": "∨",
            "not": "¬",
            "implies": "→",
            "if and only if": "↔",
            "for all": "∀",
            "there exists": "∃",
            
            # Properties and relations
            "less than": "<",
            "less than or equal": "≤",
            "greater than": ">",
            "greater than or equal": "≥",
            "equal": "=",
            "not equal": "≠",
            "divides": "∣",
            "subset": "⊆",
            "element of": "∈",
            
            # Functions and operations
            "function": "f",
            "composition": "∘",
            "identity": "id",
            "inverse": "⁻¹",
            
            # Number properties
            "even": "Even",
            "odd": "Odd",
            "prime": "Nat.Prime",
            "composite": "Nat.Composite",
            "divisible": "Nat.dvd",
            
            # Common mathematical functions
            "absolute value": "abs",
            "maximum": "max",
            "minimum": "min",
            "square root": "sqrt",
            "logarithm": "log",
            "factorial": "Nat.factorial",
            "binomial coefficient": "Nat.choose",
            
            # Calculus concepts
            "limit": "Filter.Tendsto",
            "derivative": "deriv",
            "integral": "integral",
            "continuous": "Continuous",
            "differentiable": "Differentiable"
        }
    
    def _initialize_library_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize mappings from libraries to the concepts they provide.
        
        Returns:
            Dictionary mapping libraries to concepts
        """
        return {
            "import Mathlib.Data.Nat.Basic": [
                "Nat", "Nat.add", "Nat.mul", "Nat.sub", "Nat.le", "Nat.lt", "Nat.max", "Nat.min", "Even", "Odd"
            ],
            "import Mathlib.Data.Int.Basic": [
                "Int", "Int.add", "Int.mul", "Int.sub", "Int.div", "Int.mod", "Int.abs", "Int.even", "Int.odd", "Int.gcd"
            ],
            "import Mathlib.Data.Rat.Basic": [
                "Rat", "Rat.add", "Rat.mul", "Rat.sub", "Rat.div", "Rat.num", "Rat.den"
            ],
            "import Mathlib.Data.Real.Basic": [
                "Real", "Real.add", "Real.mul", "Real.sub", "Real.div", "Real.le", "Real.lt", "Real.abs", "sqrt", "exp", "log"
            ],
            "import Mathlib.Data.Complex.Basic": [
                "Complex", "Complex.re", "Complex.im", "Complex.abs", "Complex.conjugate"
            ],
            "import Mathlib.Algebra.Group.Basic": [
                "Group", "Monoid", "Semigroup", "mul_assoc", "mul_comm", "mul_one", "one_mul"
            ],
            "import Mathlib.Algebra.Ring.Basic": [
                "Ring", "add_assoc", "add_comm", "add_zero", "zero_add", "mul_add", "add_mul"
            ],
            "import Mathlib.Algebra.Field.Basic": [
                "Field", "inv", "div", "mul_inv_cancel", "inv_mul_cancel"
            ],
            "import Mathlib.Analysis.Calculus.Deriv.Basic": [
                "deriv", "Differentiable", "differentiableAt", "hasDerivAt"
            ],
            "import Mathlib.Analysis.Calculus.Integral.Basic": [
                "integral", "integrable", "intervalIntegral"
            ],
            "import Mathlib.Topology.Basic": [
                "TopologicalSpace", "Continuous", "ContinuousAt", "Filter.Tendsto"
            ],
            "import Mathlib.Tactic.Ring": [
                "ring", "ring_nf"
            ],
            "import Mathlib.Tactic.NormNum": [
                "norm_num"
            ],
            "import Mathlib.Tactic.Linarith": [
                "linarith"
            ],
            "import Mathlib.Tactic.LibrarySearch": [
                "library_search"
            ],
            "import Mathlib.Tactic.Contrapose": [
                "contrapose", "by_contra"
            ],
            "import Mathlib.Tactic.Induction": [
                "induction"
            ],
            "import Mathlib.Tactic.Cases": [
                "cases"
            ]
        }
    
    def _initialize_notation_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from mathematical notation to Lean notation.
        
        Returns:
            Dictionary mapping notation patterns to Lean notation
        """
        return {
            # Arithmetic operations
            r'\b(\w+)\s*\+\s*(\w+)\b': r'\1 + \2',
            r'\b(\w+)\s*\-\s*(\w+)\b': r'\1 - \2',
            r'\b(\w+)\s*\*\s*(\w+)\b': r'\1 * \2',
            r'\b(\w+)\s*\/\s*(\w+)\b': r'\1 / \2',
            r'\b(\w+)\s*\^\s*(\w+)\b': r'\1 ^ \2',
            
            # Logical operations
            r'\bnot\s+(\w+)\b': r'¬ \1',
            r'\b(\w+)\s+and\s+(\w+)\b': r'\1 ∧ \2',
            r'\b(\w+)\s+or\s+(\w+)\b': r'\1 ∨ \2',
            r'\bif\s+(\w+)\s+then\s+(\w+)\b': r'\1 → \2',
            r'\b(\w+)\s+iff\s+(\w+)\b': r'\1 ↔ \2',
            
            # Comparisons
            r'\b(\w+)\s*=\s*(\w+)\b': r'\1 = \2',
            r'\b(\w+)\s*!=\s*(\w+)\b': r'\1 ≠ \2',
            r'\b(\w+)\s*<\s*(\w+)\b': r'\1 < \2',
            r'\b(\w+)\s*>\s*(\w+)\b': r'\1 > \2',
            r'\b(\w+)\s*<=\s*(\w+)\b': r'\1 ≤ \2',
            r'\b(\w+)\s*>=\s*(\w+)\b': r'\1 ≥ \2',
            
            # Quantifiers
            r'\bfor all\s+(\w+)\b': r'∀ \1,',
            r'\bthere exists\s+(\w+)\b': r'∃ \1,',
            
            # Set operations
            r'\b(\w+)\s+in\s+(\w+)\b': r'\1 ∈ \2',
            r'\b(\w+)\s+subset\s+(\w+)\b': r'\1 ⊆ \2',
            r'\b(\w+)\s+union\s+(\w+)\b': r'\1 ∪ \2',
            r'\b(\w+)\s+intersect\s+(\w+)\b': r'\1 ∩ \2',
            
            # Special notations
            r'\bdivides\((\w+),\s*(\w+)\)': r'\1 ∣ \2',
            r'\bgcd\((\w+),\s*(\w+)\)': r'Nat.gcd \1 \2',
            r'\blcm\((\w+),\s*(\w+)\)': r'Nat.lcm \1 \2',
            r'\bmax\((\w+),\s*(\w+)\)': r'max \1 \2',
            r'\bmin\((\w+),\s*(\w+)\)': r'min \1 \2'
        }
    
    def _initialize_tactic_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from abstract tactics to Lean tactics.
        
        Returns:
            Dictionary mapping tactic names to Lean tactic syntax
        """
        return {
            # Introduction tactics
            "intro": "intro {args}",
            "intros": "intros {args}",
            "assumption": "assumption",
            
            # Elimination tactics
            "destruct": "cases {args}",
            "induction": "induction {args}",
            "case_analysis": "cases {args}",
            
            # Rewriting tactics
            "rewrite": "rw [{args}]",
            "rewrite_rev": "rw [← {args}]",
            "unfold": "unfold {args}",
            "fold": "fold {args}",
            "simpl": "simp",
            
            # Automation tactics
            "auto": "tauto",
            "eauto": "tauto",
            "trivial": "trivial",
            "tauto": "tauto",
            "intuition": "tauto",
            "lia": "linarith",
            "ring": "ring",
            "field": "field_simp",
            "omega": "omega",
            
            # Existential tactics
            "exists": "use {args}",
            "econstructor": "refine ⟨_, _⟩",
            
            # Contradiction and classical tactics
            "contradiction": "contradiction",
            "absurd": "absurd {args}",
            "classical_right": "right",
            "classical_left": "left",
            
            # Application tactics
            "apply": "apply {args}",
            "eapply": "apply {args}",
            "exact": "exact {args}",
            
            # Conversion tactics
            "change": "change {args}",
            "transitivity": "transitivity {args}",
            "reflexivity": "rfl",
            "symmetry": "symm",
            
            # Structural tactics
            "split": "constructor",
            "assert": "have : {args}",
            "enough": "suffices : {args}",
            "generalize": "generalize {args}",
            "clear": "clear {args}",
            
            # Inversion tactics
            "inversion": "cases {args}",
            "discriminate": "contradiction",
            "injection": "injection {args}"
        }
    
    def _get_domain_mappings(self, domain: str) -> Dict[str, str]:
        """
        Get domain-specific concept mappings.
        
        Args:
            domain: The mathematical domain
            
        Returns:
            Dictionary mapping domain-specific concepts to Lean representations
        """
        # Domain-specific mappings
        domain_mappings = {
            # Number theory
            "11": {
                "divides": "Nat.dvd",
                "prime": "Nat.Prime",
                "even": "Even",
                "odd": "Odd",
                "gcd": "Nat.gcd",
                "lcm": "Nat.lcm",
                "coprime": "Nat.coprime",
                "mod": "Nat.mod",
                "div": "Nat.div"
            },
            
            # Algebra
            "12-20": {
                "group": "Group",
                "ring": "Ring",
                "field": "Field",
                "monoid": "Monoid",
                "vector space": "VectorSpace",
                "homomorphism": "GroupHom",
                "isomorphism": "GroupEquiv",
                "identity": "id",
                "inverse": "inv"
            },
            
            # Analysis
            "26-42": {
                "limit": "Filter.Tendsto",
                "continuous": "Continuous",
                "differentiable": "Differentiable",
                "integral": "integral",
                "derivative": "deriv",
                "sequence": "Seq",
                "series": "summable",
                "convergent": "HasLimit",
                "divergent": "¬HasLimit"
            },
            
            # Topology
            "54-55": {
                "open": "IsOpen",
                "closed": "IsClosed",
                "compact": "IsCompact",
                "connected": "IsConnected",
                "neighborhood": "nhds",
                "continuous": "Continuous",
                "homeomorphism": "Homeomorph",
                "metric space": "MetricSpace",
                "topology": "TopologicalSpace"
            }
        }
        
        # Return the mappings for the specified domain, or an empty dict if domain not found
        return domain_mappings.get(domain, {})
    
    def _get_domain_imports(self, domain: str) -> List[str]:
        """
        Get domain-specific library imports.
        
        Args:
            domain: The mathematical domain
            
        Returns:
            List of required library imports for the domain
        """
        # Domain-specific imports
        domain_imports = {
            # Number theory
            "11": [
                "import Mathlib.Data.Nat.Basic",
                "import Mathlib.Data.Nat.Prime",
                "import Mathlib.Data.Nat.GCD",
                "import Mathlib.Tactic.NormNum",
                "import Mathlib.Tactic.Ring"
            ],
            
            # Algebra
            "12-20": [
                "import Mathlib.Algebra.Group.Basic",
                "import Mathlib.Algebra.Ring.Basic",
                "import Mathlib.Algebra.Field.Basic",
                "import Mathlib.LinearAlgebra.Basic",
                "import Mathlib.Tactic.Ring"
            ],
            
            # Analysis
            "26-42": [
                "import Mathlib.Data.Real.Basic",
                "import Mathlib.Analysis.Calculus.Deriv.Basic",
                "import Mathlib.Analysis.Calculus.Integral.Basic",
                "import Mathlib.Analysis.Calculus.FDeriv.Basic",
                "import Mathlib.Analysis.Calculus.MeanValue"
            ],
            
            # Topology
            "54-55": [
                "import Mathlib.Topology.Basic",
                "import Mathlib.Topology.MetricSpace.Basic",
                "import Mathlib.Topology.Connected",
                "import Mathlib.Topology.Compact",
                "import Mathlib.Analysis.Calculus.Cont"
            ]
        }
        
        # Return the imports for the specified domain, or an empty list if domain not found
        return domain_imports.get(domain, [])


# Standalone functions for use in other modules

def map_concept_to_lean(concept: str, domain: Optional[str] = None) -> str:
    """
    Map a mathematical concept to its Lean representation.
    
    Args:
        concept: The concept to map
        domain: Optional domain for context
        
    Returns:
        The Lean representation
    """
    mapper = LeanMapper()
    return mapper.map_concept(concept, domain)

def get_lean_imports(concepts: List[str], domain: Optional[str] = None) -> List[str]:
    """
    Get required Lean library imports for a set of concepts.
    
    Args:
        concepts: List of concepts
        domain: Optional domain for context
        
    Returns:
        List of required library imports
    """
    mapper = LeanMapper()
    return mapper.get_library_imports(concepts, domain)

def map_notation_to_lean(expression: str) -> str:
    """
    Map mathematical notation to Lean notation.
    
    Args:
        expression: The mathematical expression
        
    Returns:
        The Lean notation
    """
    mapper = LeanMapper()
    return mapper.map_notation(expression)