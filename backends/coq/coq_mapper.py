"""
Coq mapper for mathematical concepts.
Maps mathematical concepts to their Coq representation.
"""

from typing import Dict, List, Any, Optional, Union
import re

class CoqMapper:
    """
    Maps mathematical concepts to their Coq representations.
    """
    
    def __init__(self):
        """Initialize the Coq mapper."""
        self.concept_mappings = self._initialize_concept_mappings()
        self.library_mappings = self._initialize_library_mappings()
        self.notation_mappings = self._initialize_notation_mappings()
        self.tactic_mappings = self._initialize_tactic_mappings()
    
    def map_concept(self, concept: str, domain: Optional[str] = None) -> str:
        """
        Map a mathematical concept to its Coq representation.
        
        Args:
            concept: The concept to map
            domain: Optional domain for context
            
        Returns:
            The Coq representation
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
        Map mathematical notation to Coq notation.
        
        Args:
            expression: The mathematical expression
            
        Returns:
            The Coq notation
        """
        result = expression
        
        # Apply notation mappings
        for pattern, replacement in self.notation_mappings.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def map_tactic(self, tactic_name: str, args: Optional[List[str]] = None) -> str:
        """
        Map a tactic name to Coq tactic syntax.
        
        Args:
            tactic_name: The tactic name
            args: Optional arguments for the tactic
            
        Returns:
            The Coq tactic syntax
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
        Initialize mappings from mathematical concepts to Coq representations.
        
        Returns:
            Dictionary mapping concepts to Coq representations
        """
        return {
            # Basic number types
            "natural number": "nat",
            "integer": "Z",
            "rational number": "Q",
            "real number": "R",
            "complex number": "C",
            
            # Mathematical structures
            "set": "Ensemble",
            "group": "Group",
            "ring": "Ring",
            "field": "Field",
            "vector space": "VectorSpace",
            
            # Logical operators
            "and": "/\\",
            "or": "\\/",
            "not": "~",
            "implies": "->",
            "if and only if": "<->",
            "for all": "forall",
            "there exists": "exists",
            
            # Properties and relations
            "less than": "<",
            "less than or equal": "<=",
            "greater than": ">",
            "greater than or equal": ">=",
            "equal": "=",
            "not equal": "<>",
            "divides": "|",
            "subset": "subset",
            "element of": "In",
            
            # Functions and operations
            "function": "f",
            "composition": "compose",
            "identity": "id",
            "inverse": "inverse",
            
            # Number properties
            "even": "even",
            "odd": "odd",
            "prime": "prime",
            "composite": "composite",
            "divisible": "divides",
            
            # Common mathematical functions
            "absolute value": "Z.abs",
            "maximum": "max",
            "minimum": "min",
            "square root": "sqrt",
            "logarithm": "log",
            "factorial": "fact",
            "binomial coefficient": "Binomial.C",
            
            # Calculus concepts
            "limit": "limit",
            "derivative": "derive",
            "integral": "integral",
            "continuous": "continuous",
            "differentiable": "differentiable"
        }
    
    def _initialize_library_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize mappings from libraries to the concepts they provide.
        
        Returns:
            Dictionary mapping libraries to concepts
        """
        return {
            "Require Import Arith.": [
                "nat", "plus", "mult", "minus", "le", "lt", "max", "min", "even", "odd"
            ],
            "Require Import ZArith.": [
                "Z", "Z.add", "Z.mul", "Z.sub", "Z.div", "Z.modulo", "Z.abs", "Z.even", "Z.odd", "Z.gcd"
            ],
            "Require Import QArith.": [
                "Q", "Qplus", "Qmult", "Qminus", "Qdiv", "Qeq", "Qlt", "Qle"
            ],
            "Require Import Reals.": [
                "R", "Rplus", "Rmult", "Rminus", "Rdiv", "Rlt", "Rle", "Rabs", "sqrt", "sin", "cos", "tan"
            ],
            "Require Import Bool.": [
                "bool", "true", "false", "andb", "orb", "negb", "if_then_else"
            ],
            "Require Import List.": [
                "list", "nil", "cons", "length", "app", "rev", "map", "fold", "filter"
            ],
            "Require Import Extraction.": [
                "extract", "Extraction"
            ],
            "Require Import Omega.": [
                "omega"
            ],
            "Require Import Lia.": [
                "lia"
            ],
            "Require Import Ring.": [
                "ring", "ring_simplify"
            ],
            "Require Import Field.": [
                "field", "field_simplify"
            ],
            "Require Import Relations.": [
                "relation", "reflexive", "symmetric", "transitive"
            ],
            "Require Import SetoidClass.": [
                "Setoid", "equivalence", "rewrite_relation"
            ],
            "Require Import Classes.Morphisms.": [
                "Proper", "respectful", "morphism"
            ],
            "Require Import Permutation.": [
                "Permutation", "perm"
            ],
            "Require Import Logic.Classical.": [
                "classic", "NNPP", "excluded_middle"
            ]
        }
    
    def _initialize_notation_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from mathematical notation to Coq notation.
        
        Returns:
            Dictionary mapping notation patterns to Coq notation
        """
        return {
            # Arithmetic operations
            r'\b(\w+)\s*\+\s*(\w+)\b': r'\1 + \2',
            r'\b(\w+)\s*\-\s*(\w+)\b': r'\1 - \2',
            r'\b(\w+)\s*\*\s*(\w+)\b': r'\1 * \2',
            r'\b(\w+)\s*\/\s*(\w+)\b': r'\1 / \2',
            r'\b(\w+)\s*\^\s*(\w+)\b': r'\1 ^ \2',
            
            # Logical operations
            r'\bnot\s+(\w+)\b': r'~ \1',
            r'\b(\w+)\s+and\s+(\w+)\b': r'\1 /\ \2',
            r'\b(\w+)\s+or\s+(\w+)\b': r'\1 \/ \2',
            r'\bif\s+(\w+)\s+then\s+(\w+)\b': r'\1 -> \2',
            r'\b(\w+)\s+iff\s+(\w+)\b': r'\1 <-> \2',
            
            # Comparisons
            r'\b(\w+)\s*=\s*(\w+)\b': r'\1 = \2',
            r'\b(\w+)\s*!=\s*(\w+)\b': r'\1 <> \2',
            r'\b(\w+)\s*<\s*(\w+)\b': r'\1 < \2',
            r'\b(\w+)\s*>\s*(\w+)\b': r'\1 > \2',
            r'\b(\w+)\s*<=\s*(\w+)\b': r'\1 <= \2',
            r'\b(\w+)\s*>=\s*(\w+)\b': r'\1 >= \2',
            
            # Quantifiers
            r'\bfor all\s+(\w+)\b': r'forall \1,',
            r'\bthere exists\s+(\w+)\b': r'exists \1,',
            
            # Set operations
            r'\b(\w+)\s+in\s+(\w+)\b': r'In \1 \2',
            r'\b(\w+)\s+subset\s+(\w+)\b': r'Included \1 \2',
            r'\b(\w+)\s+union\s+(\w+)\b': r'Union \1 \2',
            r'\b(\w+)\s+intersect\s+(\w+)\b': r'Intersection \1 \2',
            
            # Special notations
            r'\bdivides\((\w+),\s*(\w+)\)': r'divides \1 \2',
            r'\bgcd\((\w+),\s*(\w+)\)': r'gcd \1 \2',
            r'\blcm\((\w+),\s*(\w+)\)': r'lcm \1 \2',
            r'\bmax\((\w+),\s*(\w+)\)': r'max \1 \2',
            r'\bmin\((\w+),\s*(\w+)\)': r'min \1 \2'
        }
    
    def _initialize_tactic_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from abstract tactics to Coq tactics.
        
        Returns:
            Dictionary mapping tactic names to Coq tactic syntax
        """
        return {
            # Introduction tactics
            "intro": "intro {args}",
            "intros": "intros {args}",
            "assumption": "assumption",
            
            # Elimination tactics
            "destruct": "destruct {args}",
            "induction": "induction {args}",
            "case_analysis": "destruct {args}",
            
            # Rewriting tactics
            "rewrite": "rewrite {args}",
            "rewrite_rev": "rewrite <- {args}",
            "unfold": "unfold {args}",
            "fold": "fold {args}",
            "simpl": "simpl",
            
            # Automation tactics
            "auto": "auto",
            "eauto": "eauto",
            "trivial": "trivial",
            "tauto": "tauto",
            "intuition": "intuition",
            "lia": "lia",
            "ring": "ring",
            "field": "field",
            "omega": "omega",
            
            # Existential tactics
            "exists": "exists {args}",
            "econstructor": "econstructor",
            
            # Contradiction and classical tactics
            "contradiction": "contradiction",
            "absurd": "absurd {args}",
            "classical_right": "right",
            "classical_left": "left",
            
            # Application tactics
            "apply": "apply {args}",
            "eapply": "eapply {args}",
            "exact": "exact {args}",
            
            # Conversion tactics
            "change": "change {args}",
            "transitivity": "transitivity {args}",
            "reflexivity": "reflexivity",
            "symmetry": "symmetry",
            
            # Structural tactics
            "split": "split",
            "assert": "assert ({args})",
            "enough": "enough ({args})",
            "generalize": "generalize {args}",
            "clear": "clear {args}",
            
            # Inversion tactics
            "inversion": "inversion {args}",
            "discriminate": "discriminate",
            "injection": "injection {args}"
        }
    
    def _get_domain_mappings(self, domain: str) -> Dict[str, str]:
        """
        Get domain-specific concept mappings.
        
        Args:
            domain: The mathematical domain
            
        Returns:
            Dictionary mapping domain-specific concepts to Coq representations
        """
        # Domain-specific mappings
        domain_mappings = {
            # Number theory
            "11": {
                "divides": "Nat.divide",
                "prime": "prime",
                "even": "Nat.Even",
                "odd": "Nat.Odd",
                "gcd": "Nat.gcd",
                "lcm": "Nat.lcm",
                "coprime": "coprime",
                "mod": "Nat.modulo",
                "div": "Nat.div"
            },
            
            # Algebra
            "12-20": {
                "group": "Group",
                "ring": "Ring",
                "field": "Field",
                "monoid": "Monoid",
                "vector space": "VectorSpace",
                "homomorphism": "Homomorphism",
                "isomorphism": "Isomorphism",
                "identity": "ident",
                "inverse": "inverse"
            },
            
            # Analysis
            "26-42": {
                "limit": "limit",
                "continuous": "continuous",
                "differentiable": "differentiable",
                "integral": "integral",
                "derivative": "derivative",
                "sequence": "sequence",
                "series": "series",
                "convergent": "convergent",
                "divergent": "divergent"
            },
            
            # Topology
            "54-55": {
                "open": "open",
                "closed": "closed",
                "compact": "compact",
                "connected": "connected",
                "neighborhood": "neighborhood",
                "continuous": "continuous",
                "homeomorphism": "homeomorphism",
                "metric space": "metric_space",
                "topology": "topology"
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
                "Require Import Arith.",
                "Require Import ZArith.",
                "Require Import Znumtheory.",
                "Require Import Lia."
            ],
            
            # Algebra
            "12-20": [
                "Require Import Algebra.",
                "Require Import Ring.",
                "Require Import Field.",
                "Require Import LinearAlgebra."
            ],
            
            # Analysis
            "26-42": [
                "Require Import Reals.",
                "Require Import Ranalysis.",
                "Require Import Rtrigo.",
                "Require Import RiemannInt."
            ],
            
            # Topology
            "54-55": [
                "Require Import Topology.",
                "Require Import Reals.",
                "Require Import MetricSpaces."
            ]
        }
        
        # Return the imports for the specified domain, or an empty list if domain not found
        return domain_imports.get(domain, [])


# Standalone functions for use in other modules

def map_concept_to_coq(concept: str, domain: Optional[str] = None) -> str:
    """
    Map a mathematical concept to its Coq representation.
    
    Args:
        concept: The concept to map
        domain: Optional domain for context
        
    Returns:
        The Coq representation
    """
    mapper = CoqMapper()
    return mapper.map_concept(concept, domain)

def get_coq_imports(concepts: List[str], domain: Optional[str] = None) -> List[str]:
    """
    Get required Coq library imports for a set of concepts.
    
    Args:
        concepts: List of concepts
        domain: Optional domain for context
        
    Returns:
        List of required library imports
    """
    mapper = CoqMapper()
    return mapper.get_library_imports(concepts, domain)

def map_notation_to_coq(expression: str) -> str:
    """
    Map mathematical notation to Coq notation.
    
    Args:
        expression: The mathematical expression
        
    Returns:
        The Coq notation
    """
    mapper = CoqMapper()
    return mapper.map_notation(expression)