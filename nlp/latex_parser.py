"""
LaTeX parser for mathematical expressions.
Extracts structure and meaning from LaTeX math notation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger("latex_parser")

class ExpressionType(Enum):
    """Types of mathematical expressions."""
    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    LOGICAL = "logical"
    RELATION = "relation"
    SET = "set"
    QUANTIFIED = "quantified"
    UNKNOWN = "unknown"

class LatexParser:
    """
    Parser for LaTeX mathematical expressions.
    Extracts structure and semantics from LaTeX notation.
    """
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        # Known mathematical symbols
        self.symbols = {
            # Arithmetic operators
            "+": {"type": "operator", "name": "plus", "category": "arithmetic"},
            "-": {"type": "operator", "name": "minus", "category": "arithmetic"},
            "*": {"type": "operator", "name": "times", "category": "arithmetic"},
            "/": {"type": "operator", "name": "divided by", "category": "arithmetic"},
            "\\cdot": {"type": "operator", "name": "times", "category": "arithmetic"},
            "\\times": {"type": "operator", "name": "times", "category": "arithmetic"},
            "\\div": {"type": "operator", "name": "divided by", "category": "arithmetic"},
            "\\frac": {"type": "operator", "name": "fraction", "category": "arithmetic"},
            "^": {"type": "operator", "name": "power", "category": "arithmetic"},
            "_": {"type": "operator", "name": "subscript", "category": "arithmetic"},
            "\\sqrt": {"type": "operator", "name": "square root", "category": "arithmetic"},
            
            # Relational operators
            "=": {"type": "relation", "name": "equals", "category": "relation"},
            "\\neq": {"type": "relation", "name": "not equal to", "category": "relation"},
            "<": {"type": "relation", "name": "less than", "category": "relation"},
            ">": {"type": "relation", "name": "greater than", "category": "relation"},
            "\\leq": {"type": "relation", "name": "less than or equal to", "category": "relation"},
            "\\geq": {"type": "relation", "name": "greater than or equal to", "category": "relation"},
            "\\equiv": {"type": "relation", "name": "equivalent to", "category": "relation"},
            "\\approx": {"type": "relation", "name": "approximately equal to", "category": "relation"},
            "\\sim": {"type": "relation", "name": "similar to", "category": "relation"},
            "\\cong": {"type": "relation", "name": "congruent to", "category": "relation"},
            
            # Logical operators
            "\\land": {"type": "logical", "name": "and", "category": "logical"},
            "\\lor": {"type": "logical", "name": "or", "category": "logical"},
            "\\lnot": {"type": "logical", "name": "not", "category": "logical"},
            "\\Rightarrow": {"type": "logical", "name": "implies", "category": "logical"},
            "\\Leftarrow": {"type": "logical", "name": "is implied by", "category": "logical"},
            "\\Leftrightarrow": {"type": "logical", "name": "if and only if", "category": "logical"},
            "\\forall": {"type": "quantifier", "name": "for all", "category": "logical"},
            "\\exists": {"type": "quantifier", "name": "there exists", "category": "logical"},
            
            # Set operators
            "\\in": {"type": "relation", "name": "in", "category": "set"},
            "\\notin": {"type": "relation", "name": "not in", "category": "set"},
            "\\subset": {"type": "relation", "name": "subset of", "category": "set"},
            "\\subseteq": {"type": "relation", "name": "subset or equal to", "category": "set"},
            "\\supset": {"type": "relation", "name": "superset of", "category": "set"},
            "\\supseteq": {"type": "relation", "name": "superset or equal to", "category": "set"},
            "\\cup": {"type": "operator", "name": "union", "category": "set"},
            "\\cap": {"type": "operator", "name": "intersection", "category": "set"},
            "\\setminus": {"type": "operator", "name": "set minus", "category": "set"},
            "\\emptyset": {"type": "constant", "name": "empty set", "category": "set"},
            
            # Common mathematical sets
            "\\mathbb{N}": {"type": "set", "name": "natural numbers", "category": "set"},
            "\\mathbb{Z}": {"type": "set", "name": "integers", "category": "set"},
            "\\mathbb{Q}": {"type": "set", "name": "rational numbers", "category": "set"},
            "\\mathbb{R}": {"type": "set", "name": "real numbers", "category": "set"},
            "\\mathbb{C}": {"type": "set", "name": "complex numbers", "category": "set"}
        }
    
    def parse(self, latex_string: str) -> Dict[str, Any]:
        """
        Parse a LaTeX mathematical expression.
        
        Args:
            latex_string: LaTeX string representing a mathematical expression
            
        Returns:
            Dictionary with parsed information about the expression
        """
        # Clean and normalize the LaTeX string
        normalized = self._normalize_latex(latex_string)
        
        # Identify the type of expression
        expr_type = self._identify_expression_type(normalized)
        
        # Extract components of the expression
        components = self._extract_components(normalized, expr_type)
        
        # Extract variables used in the expression
        variables = self._extract_variables(normalized)
        
        # Extract operations used in the expression
        operations = self._extract_operations(normalized)
        
        # Generate a natural language interpretation
        interpretation = self._generate_interpretation(normalized, expr_type, components)
        
        return {
            "original": latex_string,
            "normalized": normalized,
            "type": expr_type.value,
            "variables": variables,
            "operations": operations,
            "components": components,
            "interpretation": interpretation
        }
    
    def _normalize_latex(self, latex_string: str) -> str:
        """
        Normalize LaTeX string by removing whitespace and fixing common issues.
        
        Args:
            latex_string: The original LaTeX string
            
        Returns:
            Normalized LaTeX string
        """
        # Remove any surrounding whitespace
        result = latex_string.strip()
        
        # Remove redundant whitespace
        result = re.sub(r'\s+', ' ', result)
        
        # Normalize spacing around operators
        for op in ['+', '-', '=', '<', '>']:
            result = re.sub(f'([^\\\\]){re.escape(op)}', f'\\1 {op} ', result)
            result = re.sub(f'{re.escape(op)}([^\\s])', f'{op} \\1', result)
        
        # Normalize fraction notation
        result = re.sub(r'\\frac\s*{([^}]*)}\s*{([^}]*)}', r'\\frac{\1}{\2}', result)
        
        return result
    
    def _identify_expression_type(self, latex: str) -> ExpressionType:
        """
        Identify the type of mathematical expression.
        
        Args:
            latex: Normalized LaTeX string
            
        Returns:
            Expression type
        """
        # Check for quantifiers (must check first since they often contain other types)
        if '\\forall' in latex or '\\exists' in latex:
            return ExpressionType.QUANTIFIED
        
        # Check for set operations
        if any(sym in latex for sym in ['\\in', '\\subset', '\\cup', '\\cap', '\\emptyset', '\\mathbb{N}', '\\mathbb{Z}']):
            return ExpressionType.SET
        
        # Check for logical operators
        if any(sym in latex for sym in ['\\land', '\\lor', '\\lnot', '\\Rightarrow', '\\Leftarrow', '\\Leftrightarrow']):
            return ExpressionType.LOGICAL
        
        # Check for relational operators
        if any(sym in latex for sym in ['=', '\\neq', '<', '>', '\\leq', '\\geq', '\\equiv', '\\approx']):
            return ExpressionType.RELATION
        
        # Check for arithmetic operations
        if any(sym in latex for sym in ['+', '-', '*', '/', '\\cdot', '\\times', '\\div', '\\frac']):
            # Check if it also has variables (algebraic) or just numbers (arithmetic)
            variables = re.findall(r'[a-zA-Z]', latex)
            if variables:
                return ExpressionType.ALGEBRAIC
            else:
                return ExpressionType.ARITHMETIC
        
        # Check if it's algebraic (has variables)
        if re.search(r'[a-zA-Z]', latex):
            return ExpressionType.ALGEBRAIC
        
        # Default to unknown
        return ExpressionType.UNKNOWN
    
    def _extract_components(self, latex: str, expr_type: ExpressionType) -> Dict[str, Any]:
        """
        Extract components from the expression based on its type.
        
        Args:
            latex: Normalized LaTeX string
            expr_type: Type of the expression
            
        Returns:
            Dictionary with components
        """
        components = {}
        
        if expr_type == ExpressionType.RELATION:
            # Try to split into left and right parts around a relation symbol
            for symbol in ['=', '\\neq', '<', '>', '\\leq', '\\geq', '\\equiv', '\\approx']:
                if symbol in latex:
                    parts = latex.split(symbol, 1)
                    if len(parts) == 2:
                        components["relation"] = symbol
                        components["left"] = parts[0].strip()
                        components["right"] = parts[1].strip()
                        break
        
        elif expr_type == ExpressionType.QUANTIFIED:
            # Extract quantifier and variable
            quantifier_match = re.search(r'(\\forall|\\exists)\s*([a-zA-Z])', latex)
            if quantifier_match:
                components["quantifier"] = quantifier_match.group(1)
                components["variable"] = quantifier_match.group(2)
                
                # Try to extract the body of the quantified expression
                var = components["variable"]
                body_match = re.search(f'{re.escape(quantifier_match.group(0))}\\s*(.*)', latex)
                if body_match:
                    components["body"] = body_match.group(1).strip()
        
        elif expr_type == ExpressionType.SET:
            # Check for specific set operations
            if '\\in' in latex:
                parts = latex.split('\\in', 1)
                if len(parts) == 2:
                    components["operation"] = "\\in"
                    components["element"] = parts[0].strip()
                    components["set"] = parts[1].strip()
            
            elif '\\subset' in latex:
                parts = latex.split('\\subset', 1)
                if len(parts) == 2:
                    components["operation"] = "\\subset"
                    components["subset"] = parts[0].strip()
                    components["superset"] = parts[1].strip()
            
            elif '\\cup' in latex:
                parts = latex.split('\\cup', 1)
                if len(parts) == 2:
                    components["operation"] = "\\cup"
                    components["set1"] = parts[0].strip()
                    components["set2"] = parts[1].strip()
            
            elif '\\cap' in latex:
                parts = latex.split('\\cap', 1)
                if len(parts) == 2:
                    components["operation"] = "\\cap"
                    components["set1"] = parts[0].strip()
                    components["set2"] = parts[1].strip()
        
        elif expr_type in [ExpressionType.ARITHMETIC, ExpressionType.ALGEBRAIC]:
            # Extract terms for simple expressions
            # This is a simplification; real parsing of complex expressions would be more involved
            if '+' in latex:
                components["operation"] = "addition"
                components["terms"] = [term.strip() for term in latex.split('+')]
            
            elif '-' in latex:
                # Check if it's a subtraction or a negative number
                if latex.startswith('-'):
                    components["operation"] = "negation"
                    components["term"] = latex[1:].strip()
                else:
                    components["operation"] = "subtraction"
                    components["terms"] = [term.strip() for term in latex.split('-', 1)]
            
            elif '*' in latex or '\\cdot' in latex or '\\times' in latex:
                delimiter = '*' if '*' in latex else ('\\cdot' if '\\cdot' in latex else '\\times')
                components["operation"] = "multiplication"
                components["terms"] = [term.strip() for term in latex.split(delimiter)]
            
            elif '/' in latex or '\\div' in latex:
                delimiter = '/' if '/' in latex else '\\div'
                components["operation"] = "division"
                parts = latex.split(delimiter, 1)
                if len(parts) == 2:
                    components["numerator"] = parts[0].strip()
                    components["denominator"] = parts[1].strip()
            
            elif '\\frac' in latex:
                frac_match = re.search(r'\\frac\{([^}]*)\}\{([^}]*)\}', latex)
                if frac_match:
                    components["operation"] = "fraction"
                    components["numerator"] = frac_match.group(1).strip()
                    components["denominator"] = frac_match.group(2).strip()
        
        return components
    
    def _extract_variables(self, latex: str) -> List[str]:
        """
        Extract variables from a LaTeX expression.
        
        Args:
            latex: Normalized LaTeX string
            
        Returns:
            List of variables
        """
        # Extract single letter variables (possibly with subscripts)
        var_pattern = r'([a-zA-Z])(?:_\{?([a-zA-Z0-9]+)\}?)?'
        matches = re.finditer(var_pattern, latex)
        
        variables = []
        for match in matches:
            var = match.group(1)
            if match.group(2):  # Has subscript
                var = f"{var}_{match.group(2)}"
            
            # Skip if it's part of a LaTeX command
            if f"\\{match.group(0)}" not in latex and f"\\{match.group(1)}" not in latex:
                variables.append(var)
        
        return list(set(variables))
    
    def _extract_operations(self, latex: str) -> List[Dict[str, str]]:
        """
        Extract operations from a LaTeX expression.
        
        Args:
            latex: Normalized LaTeX string
            
        Returns:
            List of operations with their properties
        """
        operations = []
        
        # Check for each known symbol
        for symbol, info in self.symbols.items():
            if symbol in latex:
                operations.append({
                    "symbol": symbol,
                    "name": info["name"],
                    "type": info["type"],
                    "category": info["category"]
                })
        
        return operations
    
    def _generate_interpretation(self, latex: str, expr_type: ExpressionType, components: Dict[str, Any]) -> str:
        """
        Generate a natural language interpretation of the expression.
        
        Args:
            latex: Normalized LaTeX string
            expr_type: Type of the expression
            components: Components of the expression
            
        Returns:
            Natural language interpretation
        """
        if expr_type == ExpressionType.RELATION:
            if "relation" in components and "left" in components and "right" in components:
                relation = components["relation"]
                left = components["left"]
                right = components["right"]
                
                relation_text = self.symbols.get(relation, {}).get("name", relation)
                return f"{left} {relation_text} {right}"
        
        elif expr_type == ExpressionType.QUANTIFIED:
            if "quantifier" in components and "variable" in components:
                quantifier = components["quantifier"]
                variable = components["variable"]
                body = components.get("body", "")
                
                quantifier_text = self.symbols.get(quantifier, {}).get("name", quantifier)
                return f"{quantifier_text} {variable}, {body}"
        
        elif expr_type == ExpressionType.SET:
            if "operation" in components:
                operation = components["operation"]
                
                if operation == "\\in" and "element" in components and "set" in components:
                    element = components["element"]
                    set_expr = components["set"]
                    return f"{element} is an element of {set_expr}"
                
                elif operation == "\\subset" and "subset" in components and "superset" in components:
                    subset = components["subset"]
                    superset = components["superset"]
                    return f"{subset} is a subset of {superset}"
                
                elif operation == "\\cup" and "set1" in components and "set2" in components:
                    set1 = components["set1"]
                    set2 = components["set2"]
                    return f"the union of {set1} and {set2}"
                
                elif operation == "\\cap" and "set1" in components and "set2" in components:
                    set1 = components["set1"]
                    set2 = components["set2"]
                    return f"the intersection of {set1} and {set2}"
        
        elif expr_type in [ExpressionType.ARITHMETIC, ExpressionType.ALGEBRAIC]:
            if "operation" in components:
                operation = components["operation"]
                
                if operation == "addition" and "terms" in components:
                    terms = components["terms"]
                    return f"the sum of {' and '.join(terms)}"
                
                elif operation == "subtraction" and "terms" in components:
                    terms = components["terms"]
                    if len(terms) == 2:
                        return f"{terms[0]} minus {terms[1]}"
                
                elif operation == "multiplication" and "terms" in components:
                    terms = components["terms"]
                    return f"the product of {' and '.join(terms)}"
                
                elif operation in ["division", "fraction"] and "numerator" in components and "denominator" in components:
                    numerator = components["numerator"]
                    denominator = components["denominator"]
                    return f"{numerator} divided by {denominator}"
                
                elif operation == "negation" and "term" in components:
                    term = components["term"]
                    return f"the negative of {term}"
        
        # Default to the original LaTeX if we can't interpret it
        return f"the expression {latex}"