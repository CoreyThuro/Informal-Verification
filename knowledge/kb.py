"""
Simplified knowledge base for the proof translator system.
"""

import json
import os
from typing import Dict, List, Any, Optional

class KnowledgeBase:
    """
    Lightweight knowledge base for mathematical domains and proof patterns.
    """
    
    def __init__(self, data_dir="knowledge/data"):
        """
        Initialize the knowledge base.
        
        Args:
            data_dir: Directory containing knowledge data files
        """
        self.data_dir = data_dir
        
        # Load knowledge data
        self.domains = self._load_json("domains.json")
        self.patterns = self._load_json("patterns.json")
        self.tactics = self._load_json("tactics.json")
        
        # Initialize semantic knowledge structures
        self._init_semantic_knowledge()
        
        print(f"Loaded knowledge base with {len(self.domains)} domains and {len(self.patterns)} patterns")
    
    def _load_json(self, filename: str) -> Dict:
        """Load data from a JSON file."""
        path = os.path.join(self.data_dir, filename)
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty dict if file not found or invalid
            return {}
    
    def get_domain_info(self, domain_code: str) -> Dict[str, Any]:
        """Get information about a mathematical domain."""
        if domain_code in self.domains:
            return self.domains[domain_code]
        return {}
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get information about a proof pattern."""
        if pattern_name in self.patterns:
            return self.patterns[pattern_name]
        return {}
    
    def get_domain_tactics(self, domain_code: str) -> List[Dict[str, str]]:
        """Get recommended tactics for a domain."""
        if domain_code in self.tactics.get("domains", {}):
            return self.tactics["domains"][domain_code]
        return []
    
    def get_pattern_tactics(self, pattern_name: str) -> List[Dict[str, str]]:
        """Get recommended tactics for a pattern."""
        if pattern_name in self.tactics.get("patterns", {}):
            return self.tactics["patterns"][pattern_name]
        
        # Fallback default tactics for common patterns if not found in the knowledge base
        default_tactics = {
            "evenness": [
                {"name": "intros", "description": "Introduce variables"},
                {"name": "exists", "description": "Provide witness for existential"},
                {"name": "ring", "description": "Solve algebraic equality"}
            ],
            "induction": [
                {"name": "intros", "description": "Introduce variables"},
                {"name": "induction", "description": "Apply induction principle"},
                {"name": "simpl", "description": "Simplify expressions"},
                {"name": "rewrite", "description": "Rewrite using hypothesis"}
            ],
            "contradiction": [
                {"name": "intros", "description": "Introduce variables"},
                {"name": "intro", "description": "Introduce negation"},
                {"name": "assert", "description": "Make assertion"},
                {"name": "contradiction", "description": "Derive contradiction"}
            ],
            "cases": [
                {"name": "intros", "description": "Introduce variables"},
                {"name": "destruct", "description": "Case analysis"},
                {"name": "simpl", "description": "Simplify expressions"}
            ],
            "direct": [
                {"name": "intros", "description": "Introduce variables"},
                {"name": "unfold", "description": "Expand definitions"},
                {"name": "apply", "description": "Apply theorem"}
            ]
        }
        
        return default_tactics.get(pattern_name, [])
    
    def get_imports_for_domain(self, domain_code: str) -> List[str]:
        """Get required Coq imports for a domain."""
        imports = []
        
        # Core imports that are always needed
        imports.append("Require Import Arith.")
        
        # Domain-specific imports
        if domain_code == "11":  # Number theory
            imports.extend([
                "Require Import Lia.",
                "Require Import ZArith.",
                "Require Import Znumtheory.",  # Advanced number theory
                "Require Import Zdiv."         # Division properties
            ])
        elif domain_code in ["12", "13", "14", "15", "16", "17", "18", "19", "20"]:  # Algebra
            imports.extend([
                "Require Import Ring.",
                "Require Import Field.",
                "Require Import Algebra."  # If available in the Coq installation
            ])
        elif domain_code in ["26", "27", "28", "30", "31", "32", "33", "34", "35"]:  # Analysis
            imports.extend([
                "Require Import Reals.",
                "Require Import Ranalysis."  # Real analysis
            ])
        elif domain_code in ["54", "55"]:  # Topology
            imports.extend([
                "Require Import Reals.",
                "Require Import Topology."  # If available in the Coq installation
            ])
            
        # Get additional imports from the knowledge base
        domain_info = self.get_domain_info(domain_code)
        if "imports" in domain_info:
            imports.extend(domain_info["imports"])
        
        # Deduplicate and filter out any imports that might not be available
        # This is a simplified approach - in a real system, you'd check if the imports exist
        standard_imports = [
            "Require Import Arith.",
            "Require Import Lia.",
            "Require Import ZArith.",
            "Require Import Znumtheory.",
            "Require Import Zdiv.",
            "Require Import Ring.",
            "Require Import Field.",
            "Require Import Reals.",
            "Require Import Ranalysis."
        ]
        
        filtered_imports = [imp for imp in imports if imp in standard_imports]
        if not filtered_imports:  # If all were filtered out, return the core ones
            filtered_imports = ["Require Import Arith.", "Require Import Lia."]
            
        return list(dict.fromkeys(filtered_imports))
    
    def get_tactic_string(self, tactic_name: str, args: Optional[List[str]] = None) -> str:
        """Get the Coq syntax for a tactic."""
        args_str = " ".join(args) if args else ""
        
        # Map tactic names to Coq syntax
        tactic_map = {
            "intro": f"intro {args_str}",
            "intros": f"intros {args_str}",
            "apply": f"apply {args_str}",
            "rewrite": f"rewrite {args_str}",
            "destruct": f"destruct {args_str}",
            "induction": f"induction {args_str}",
            "exists": f"exists {args_str}",
            "ring": "ring",
            "lia": "lia",
            "auto": "auto",
            "simpl": "simpl",
            "reflexivity": "reflexivity",
            "contradiction": "contradiction",
            # Advanced tactics
            "assert": f"assert ({args_str})",
            "pose": f"pose ({args_str})",
            "unfold": f"unfold {args_str}",
            "fold": f"fold {args_str}",
            "case": f"case {args_str}",
            "clear": f"clear {args_str}",
            "specialize": f"specialize {args_str}",
            "generalize": f"generalize {args_str}",
            "remember": f"remember {args_str}",
            "subst": "subst",
            "discriminate": "discriminate",
            "injection": f"injection {args_str}",
            "split": "split",
            "left": "left",
            "right": "right",
            "firstorder": "firstorder",
            "tauto": "tauto",
            "congruence": "congruence"
        }
        
        return tactic_map.get(tactic_name, f"{tactic_name} {args_str}")
    
    def _init_semantic_knowledge(self):
        """
        Initialize semantic knowledge structures for enhanced pattern recognition.
        """
        # Mathematical domain keywords with expanded vocabulary
        self.domain_keywords = {
            "11": [  # Number Theory
                "prime", "number", "integer", "divisible", "even", "odd", "gcd", "lcm",
                "divisor", "factor", "multiple", "remainder", "modulo", "congruent",
                "coprime", "relatively prime", "parity", "divides", "quotient",
                "natural number", "rational", "irrational", "composite", "perfect number"
            ],
            "12-20": [  # Algebra
                "group", "ring", "field", "algebra", "vector", "matrix", "linear", 
                "polynomial", "homomorphism", "isomorphism", "commutative", "associative",
                "distributive", "identity", "inverse", "subgroup", "ideal", "kernel",
                "eigenvalue", "eigenvector", "determinant", "trace", "basis", "dimension",
                "linear transformation", "span", "linearly independent", "nullspace"
            ],
            "26-42": [  # Analysis
                "limit", "continuous", "derivative", "integral", "function", "series", 
                "sequence", "convergent", "divergent", "differentiable", "bounded",
                "uniform", "pointwise", "cauchy", "complete", "metric", "norm",
                "supremum", "infimum", "maximum", "minimum", "neighborhood", "open ball",
                "closed ball", "epsilon", "delta", "taylor series", "power series"
            ],
            "54-55": [  # Topology
                "topology", "open", "closed", "compact", "connected", "neighborhood", 
                "homeomorphic", "metric", "space", "continuous", "hausdorff",
                "cover", "subcover", "finite subcover", "basis", "subbasis", "interior",
                "closure", "boundary", "dense", "separable", "first countable",
                "second countable", "locally compact", "paracompact", "manifold"
            ]
        }
        
        # Proof pattern indicators with expanded vocabulary
        self.pattern_indicators = {
            "evenness": [
                "even", "divisible by 2", "multiple of 2", "divisible by two",
                "x + x", "2x", "2 * x", "parity", "odd", "remainder when divided by 2"
            ],
            "induction": [
                "induction", "base case", "inductive step", "inductive hypothesis",
                "for n = 0", "assume for k", "prove for k+1", "by induction",
                "mathematical induction", "strong induction", "complete induction",
                "principle of induction", "induction on", "inductive proof"
            ],
            "contradiction": [
                "contradiction", "assume not", "suppose not", "contrary", "absurd",
                "impossible", "leads to a contradiction", "proof by contradiction",
                "assume the opposite", "assume the contrary", "for the sake of contradiction",
                "reductio ad absurdum", "derive a contradiction", "contradicts"
            ],
            "cases": [
                "case", "cases", "first case", "second case", "consider the case",
                "split into cases", "by cases", "case analysis", "in the case where",
                "either", "or", "when", "if", "otherwise", "divide into cases",
                "separate into cases", "consider separately", "in each case"
            ],
            "direct": [
                "direct proof", "straightforward", "directly", "immediately",
                "follows from", "by definition", "using", "applying", "since",
                "because", "therefore", "thus", "hence", "so", "we have",
                "we get", "we obtain", "this gives", "this implies"
            ]
        }
        
        # Mathematical symbols and notations
        self.mathematical_symbols = {
            "arithmetic": ["+", "-", "*", "/", "^", "√", "∑", "∏", "mod", "div"],
            "logical": ["∧", "∨", "¬", "→", "↔", "⊕", "⊻"],
            "relational": ["=", "≠", "<", ">", "≤", "≥", "≈", "≡", "≢", "∝"],
            "set": ["∈", "∉", "⊂", "⊃", "⊆", "⊇", "∪", "∩", "∅", "\\"]
        }
        
        # Common mathematical structures and their properties
        self.mathematical_structures = {
            "group": ["identity", "inverse", "associative", "abelian", "commutative", "order"],
            "ring": ["additive identity", "multiplicative identity", "distributive"],
            "field": ["multiplicative inverse", "division"],
            "vector_space": ["basis", "dimension", "span", "linear independence"],
            "topology": ["open sets", "closed sets", "continuous functions"]
        }
    
    def get_domain_keywords(self) -> Dict[str, List[str]]:
        """
        Get the domain keywords for domain detection.
        
        Returns:
            Dictionary mapping domain codes to lists of keywords
        """
        return self.domain_keywords if hasattr(self, 'domain_keywords') else {
            "11": ["prime", "number", "integer", "divisible", "even", "odd", "gcd"],
            "12-20": ["group", "ring", "field", "algebra", "vector", "matrix"],
            "26-42": ["limit", "continuous", "derivative", "integral", "function"],
            "54-55": ["topology", "open", "closed", "compact", "connected"]
        }
    
    def get_pattern_indicators(self, pattern_name: str = None) -> Dict[str, List[str]]:
        """
        Get indicators for proof patterns.
        
        Args:
            pattern_name: Optional specific pattern to get indicators for
            
        Returns:
            Dictionary mapping pattern names to lists of indicators,
            or list of indicators for a specific pattern if pattern_name is provided
        """
        if not hasattr(self, 'pattern_indicators'):
            return {}
            
        if pattern_name:
            return self.pattern_indicators.get(pattern_name, [])
        return self.pattern_indicators
    
    def analyze_theorem_structure(self, theorem_text: str) -> Dict[str, Any]:
        """
        Analyze the structure of a theorem statement.
        
        Args:
            theorem_text: The theorem statement
            
        Returns:
            Dictionary with analysis results
        """
        # Simplified analysis - in a real system, this would be more sophisticated
        result = {
            "quantifiers": [],
            "variables": [],
            "conditions": [],
            "conclusion": ""
        }
        
        # Extract universal quantifiers
        if re.search(r'\b(for all|for every|for any|\\forall)\b', theorem_text, re.IGNORECASE):
            result["quantifiers"].append("universal")
        
        # Extract existential quantifiers
        if re.search(r'\b(there exists|for some|\\exists)\b', theorem_text, re.IGNORECASE):
            result["quantifiers"].append("existential")
        
        # Extract variables (simplified)
        variables = set(re.findall(r'\b([a-z])\b', theorem_text.lower()))
        result["variables"] = list(variables)
        
        # Split into condition and conclusion (very simplified)
        if ", then " in theorem_text:
            parts = theorem_text.split(", then ")
            result["conditions"] = [parts[0]]
            result["conclusion"] = parts[1]
        elif " if " in theorem_text:
            parts = theorem_text.split(" if ")
            result["conclusion"] = parts[0]
            result["conditions"] = [parts[1]]
        else:
            result["conclusion"] = theorem_text
        
        return result