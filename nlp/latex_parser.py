"""
LaTeX parser for mathematical expressions.
Provides basic parsing of LaTeX expressions in mathematical proofs.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

class LatexParser:
    """
    Parser for LaTeX mathematical expressions.
    
    This class provides functionality to parse and analyze LaTeX mathematical 
    expressions commonly found in proofs.
    """
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        # Define common LaTeX commands and symbols
        self.math_commands = {
            "\\frac": "fraction",
            "\\sqrt": "square_root",
            "\\sum": "summation",
            "\\prod": "product",
            "\\int": "integral",
            "\\lim": "limit",
            "\\inf": "infimum",
            "\\sup": "supremum",
            "\\max": "maximum",
            "\\min": "minimum",
            "\\log": "logarithm",
            "\\ln": "natural_logarithm",
            "\\sin": "sine",
            "\\cos": "cosine",
            "\\tan": "tangent",
            "\\cot": "cotangent",
            "\\sec": "secant",
            "\\csc": "cosecant",
            "\\forall": "for_all",
            "\\exists": "exists",
            "\\nexists": "not_exists",
            "\\in": "element_of",
            "\\subset": "subset",
            "\\subseteq": "subset_or_equal",
            "\\supset": "superset",
            "\\supseteq": "superset_or_equal",
            "\\cup": "union",
            "\\cap": "intersection",
            "\\setminus": "set_minus",
            "\\implies": "implies",
            "\\iff": "if_and_only_if",
            "\\neg": "negation",
            "\\land": "logical_and",
            "\\lor": "logical_or",
            "\\oplus": "xor",
            "\\equiv": "equivalent",
            "\\approx": "approximately_equal",
            "\\cong": "congruent",
            "\\sim": "similar",
            "\\neq": "not_equal",
            "\\leq": "less_than_or_equal",
            "\\geq": "greater_than_or_equal",
            "\\ll": "much_less_than",
            "\\gg": "much_greater_than",
            "\\prec": "precedes",
            "\\succ": "succeeds",
            "\\cdot": "multiplication",
            "\\times": "multiplication",
            "\\div": "division",
            "\\pm": "plus_or_minus",
            "\\mp": "minus_or_plus",
            "\\partial": "partial_derivative",
            "\\nabla": "nabla",
            "\\infty": "infinity",
            "\\aleph": "aleph",
            "\\emptyset": "empty_set",
            "\\varnothing": "empty_set",
            "\\mathbb{N}": "natural_numbers",
            "\\mathbb{Z}": "integers",
            "\\mathbb{Q}": "rational_numbers",
            "\\mathbb{R}": "real_numbers",
            "\\mathbb{C}": "complex_numbers"
        }
        
        # Common LaTeX environments
        self.math_environments = [
            "equation", "align", "gather", "multline", "eqnarray",
            "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
            "array", "cases"
        ]
    
    def parse(self, latex_text: str) -> Dict[str, Any]:
        """
        Parse a LaTeX mathematical expression.
        
        Args:
            latex_text: The LaTeX expression to parse
            
        Returns:
            Dictionary with parsed information
        """
        try:
            # Extract structure and components
            structure = self._extract_structure(latex_text)
            variables = self._extract_variables(latex_text)
            operations = self._extract_operations(latex_text)
            commands = self._extract_commands(latex_text)
            
            # Determine the expression type
            expr_type = self._determine_expression_type(structure, commands)
            
            # Extract any number literals
            numbers = self._extract_numbers(latex_text)
            
            return {
                "original": latex_text,
                "variables": list(variables),
                "operations": operations,
                "commands": commands,
                "structure": structure,
                "expression_type": expr_type,
                "numbers": numbers
            }
        except Exception as e:
            logger.warning(f"Error parsing LaTeX expression: {e}")
            return {
                "original": latex_text,
                "variables": [],
                "operations": [],
                "commands": [],
                "structure": "unknown",
                "expression_type": "unknown",
                "numbers": [],
                "error": str(e)
            }
    
    def normalize(self, latex_text: str) -> str:
        """
        Normalize a LaTeX expression for easier processing.
        
        Args:
            latex_text: The LaTeX expression to normalize
            
        Returns:
            Normalized expression
        """
        # Remove whitespace
        normalized = re.sub(r'\s+', ' ', latex_text).strip()
        
        # Convert display math mode to inline
        normalized = re.sub(r'\$\$(.*?)\$\$', r'$\1$', normalized)
        
        # Normalize fractions (handle \frac{a}{b} and {a \over b})
        normalized = re.sub(r'{([^{}]*)\s*\\over\s*([^{}]*)}', r'\\frac{\1}{\2}', normalized)
        
        # Normalize spacing around operators
        for op in ['+', '-', '=', '<', '>', r'\times', r'\div', r'\cdot']:
            normalized = re.sub(f'({op})', r' \1 ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _extract_structure(self, latex_text: str) -> str:
        """
        Extract the structure of a LaTeX expression.
        
        Args:
            latex_text: The LaTeX expression
            
        Returns:
            Structure type as a string
        """
        # Check for common structures
        if "\\frac" in latex_text:
            return "fraction"
        elif "=" in latex_text:
            return "equation"
        elif any(op in latex_text for op in [r"\leq", r"\geq", r"<", r">"]):
            return "inequality"
        elif any(op in latex_text for op in [r"\in", r"\subset", r"\subseteq"]):
            return "set_relation"
        elif any(func in latex_text for func in [r"\sum", r"\prod", r"\int"]):
            return "calculus"
        elif r"\forall" in latex_text or r"\exists" in latex_text:
            return "quantified"
        elif r"\iff" in latex_text or r"\Rightarrow" in latex_text or r"\Leftarrow" in latex_text:
            return "logical_statement"
        else:
            return "expression"
    
    def _extract_variables(self, latex_text: str) -> Set[str]:
        """
        Extract variables from a LaTeX expression.
        
        Args:
            latex_text: The LaTeX expression
            
        Returns:
            Set of variables
        """
        variables = set()
        
        # Single letter variables (most common case)
        var_matches = re.findall(r'(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])', latex_text)
        variables.update(var_matches)
        
        # Variables with subscripts like x_i, y_j
        subscript_matches = re.findall(r'([a-zA-Z])_([a-zA-Z0-9])', latex_text)
        for match in subscript_matches:
            variables.add(f"{match[0]}_{match[1]}")
            variables.add(match[0])  # Also add the base variable
        
        # Variables with custom commands like \alpha, \beta
        greek_letters = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", 
                         r"\zeta", r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa", 
                         r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\varpi", r"\rho", 
                         r"\varrho", r"\sigma", r"\varsigma", r"\tau", r"\upsilon", r"\phi", 
                         r"\varphi", r"\chi", r"\psi", r"\omega"]
        
        for letter in greek_letters:
            if letter in latex_text:
                variables.add(letter)
        
        return variables
    
    def _extract_operations(self, latex_text: str) -> List[str]:
        """
        Extract mathematical operations from a LaTeX expression.
        
        Args:
            latex_text: The LaTeX expression
            
        Returns:
            List of operations
        """
        operations = []
        
        # Check for basic operations
        if "+" in latex_text:
            operations.append("addition")
        if "-" in latex_text:
            operations.append("subtraction")
        if r"\cdot" in latex_text or r"\times" in latex_text or "*" in latex_text:
            operations.append("multiplication")
        if "/" in latex_text or r"\div" in latex_text or r"\frac" in latex_text:
            operations.append("division")
        if "^" in latex_text:
            operations.append("exponentiation")
        if "_" in latex_text:
            operations.append("subscript")
        if "=" in latex_text:
            operations.append("equality")
        if "<" in latex_text or r"\lt" in latex_text:
            operations.append("less_than")
        if ">" in latex_text or r"\gt" in latex_text:
            operations.append("greater_than")
        if r"\leq" in latex_text:
            operations.append("less_than_or_equal")
        if r"\geq" in latex_text:
            operations.append("greater_than_or_equal")
        
        return operations
    
    def _extract_commands(self, latex_text: str) -> List[str]:
        """
        Extract LaTeX commands from a LaTeX expression.
        
        Args:
            latex_text: The LaTeX expression
            
        Returns:
            List of command types
        """
        commands = []
        
        # Check for known commands
        for command, command_type in self.math_commands.items():
            if command in latex_text:
                commands.append(command_type)
        
        return commands
    
    def _determine_expression_type(self, structure: str, commands: List[str]) -> str:
        """
        Determine the type of mathematical expression.
        
        Args:
            structure: Expression structure
            commands: List of LaTeX commands
            
        Returns:
            Expression type
        """
        # Check for specific expression types based on structure and commands
        if structure == "equation":
            return "equation"
        elif structure == "inequality":
            return "inequality"
        elif structure == "set_relation":
            return "set_theory"
        elif structure == "quantified":
            return "logic"
        elif structure == "logical_statement":
            return "logic"
        elif structure == "fraction":
            return "fraction"
        elif "integral" in commands:
            return "calculus"
        elif "limit" in commands:
            return "calculus"
        elif "summation" in commands:
            return "series"
        elif "product" in commands:
            return "product"
        elif "for_all" in commands or "exists" in commands:
            return "predicate_logic"
        else:
            # Default to algebraic if it has variables and operations
            return "algebraic"
    
    def _extract_numbers(self, latex_text: str) -> List[str]:
        """
        Extract number literals from a LaTeX expression.
        
        Args:
            latex_text: The LaTeX expression
            
        Returns:
            List of number literals
        """
        # Match integers and decimals
        numbers = re.findall(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)', latex_text)
        
        # Match fractions written as \frac{a}{b}
        frac_matches = re.findall(r'\\frac\{(\d+)\}\{(\d+)\}', latex_text)
        for num, den in frac_matches:
            numbers.append(f"{num}/{den}")
        
        return numbers