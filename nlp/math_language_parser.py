"""
Mathematical language parser for handling mixed text and formulas.
Separates and processes mathematical notation within natural language.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import spacy

# Configure logging
logger = logging.getLogger("math_parser")

# Try to load SymPy for formula parsing
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, TokenError
    HAS_SYMPY = True
except ImportError:
    logger.warning("SymPy not available. Formula parsing will be limited.")
    HAS_SYMPY = False

class LatexParser:
    """
    Parser for LaTeX mathematical expressions.
    """
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        self.pattern_map = {
            # Basic arithmetic
            r'\\frac\{([^{}]+)\}\{([^{}]+)\}': self._parse_fraction,
            r'([^_^]+)_\{([^{}]+)\}': self._parse_subscript,
            r'([^_^]+)\^\{([^{}]+)\}': self._parse_superscript,
            r'\\sqrt\{([^{}]+)\}': self._parse_sqrt,
            r'\\sum_\{([^{}]+)\}\^\{([^{}]+)\}': self._parse_sum,
            
            # Sets and logic
            r'\\forall': 'for all',
            r'\\exists': 'there exists',
            r'\\in': 'in',
            r'\\subset': 'subset of',
            r'\\cup': 'union',
            r'\\cap': 'intersection',
            r'\\setminus': 'set minus',
            r'\\Rightarrow': 'implies',
            r'\\Leftrightarrow': 'if and only if',
            r'\\land': 'and',
            r'\\lor': 'or',
            r'\\lnot': 'not',
            
            # Number theory
            r'\\mathbb\{N\}': 'natural numbers',
            r'\\mathbb\{Z\}': 'integers',
            r'\\mathbb\{Q\}': 'rational numbers',
            r'\\mathbb\{R\}': 'real numbers',
            r'\\mathbb\{C\}': 'complex numbers',
            r'\\mid': 'divides',
            r'\\equiv': 'congruent to',
            r'\\gcd': 'gcd',
            r'\\lcm': 'lcm'
        }
    
    def parse(self, latex: str) -> Dict[str, Any]:
        """
        Parse a LaTeX mathematical expression.
        
        Args:
            latex: The LaTeX expression string
            
        Returns:
            Dictionary with parsed information
        """
        # Basic text normalization
        latex = latex.strip()
        
        # Default parsed result
        result = {
            "original": latex,
            "parsed_text": latex,
            "expression_type": "unknown",
            "variables": [],
            "operation": None,
            "sympy_expr": None
        }
        
        # Extract variables
        variables = re.findall(r'\\mathit\{([a-zA-Z])\}|\\mathrm\{([a-zA-Z])\}|(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])', latex)
        result["variables"] = list(set(v[0] or v[1] or v[2] for v in variables if v[0] or v[1] or v[2]))
        
        # Apply pattern matching
        parsed_text = latex
        for pattern, handler in self.pattern_map.items():
            if callable(handler):
                match = re.search(pattern, parsed_text)
                if match:
                    parsing_result = handler(match)
                    if isinstance(parsing_result, dict):
                        # Update result with handler's information
                        result.update(parsing_result)
                    else:
                        # Replace with handler's parsed text
                        parsed_text = re.sub(pattern, str(parsing_result), parsed_text)
            else:
                # Direct replacement
                parsed_text = re.sub(pattern, handler, parsed_text)
        
        # Update parsed text
        result["parsed_text"] = parsed_text
        
        # Try to parse with SymPy
        if HAS_SYMPY:
            try:
                # Create symbols for variables
                for var in result["variables"]:
                    exec(f"{var} = sympy.symbols('{var}')")
                
                # Try to parse with SymPy
                sympy_expr = parse_expr(parsed_text.replace('\\', ''))
                result["sympy_expr"] = str(sympy_expr)
                
                # Determine expression type
                if isinstance(sympy_expr, sympy.Eq):
                    result["expression_type"] = "equation"
                    result["operation"] = "equals"
                elif isinstance(sympy_expr, sympy.Rel):
                    result["expression_type"] = "relation"
                    result["operation"] = str(type(sympy_expr).__name__).lower()
                elif isinstance(sympy_expr, sympy.Add):
                    result["expression_type"] = "addition"
                    result["operation"] = "plus"
                elif isinstance(sympy_expr, sympy.Mul):
                    result["expression_type"] = "multiplication"
                    result["operation"] = "times"
                elif isinstance(sympy_expr, sympy.Pow):
                    result["expression_type"] = "exponentiation"
                    result["operation"] = "power"
                else:
                    result["expression_type"] = "expression"
            except (TokenError, SyntaxError, NameError) as e:
                logger.debug(f"Failed to parse with SymPy: {e}")
        
        return result
    
    def _parse_fraction(self, match) -> str:
        """Parse a LaTeX fraction."""
        numerator = match.group(1)
        denominator = match.group(2)
        return f"({numerator}) / ({denominator})"
    
    def _parse_subscript(self, match) -> str:
        """Parse a LaTeX subscript."""
        base = match.group(1)
        subscript = match.group(2)
        return f"{base}_{subscript}"
    
    def _parse_superscript(self, match) -> str:
        """Parse a LaTeX superscript."""
        base = match.group(1)
        exponent = match.group(2)
        return f"{base}^{exponent}"
    
    def _parse_sqrt(self, match) -> str:
        """Parse a LaTeX square root."""
        content = match.group(1)
        return f"sqrt({content})"
    
    def _parse_sum(self, match) -> Dict[str, Any]:
        """Parse a LaTeX summation."""
        lower = match.group(1)
        upper = match.group(2)
        return {
            "expression_type": "summation",
            "operation": "sum",
            "lower_bound": lower,
            "upper_bound": upper
        }

class MathLanguageParser:
    """
    Parser for mathematical language with mixed text and formulas.
    """
    
    def __init__(self):
        """Initialize the mathematical language parser."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install with 'python -m spacy download en_core_web_sm'")
            self.nlp = None
        
        self.latex_parser = LatexParser()
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse mathematical text with mixed natural language and formulas.
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary with parsed information
        """
        # Ensure we have a working NLP model
        if self.nlp is None:
            raise ValueError("No spaCy model available. Cannot parse text.")
        
        # Identify LaTeX segments
        latex_segments = self._extract_latex_segments(text)
        
        # Replace LaTeX with placeholders for NLP
        nlp_text, placeholders = self._replace_latex(text, latex_segments)
        
        # Parse with spaCy
        doc = self.nlp(nlp_text)
        
        # Extract basic NLP information
        nlp_info = self._extract_nlp_info(doc)
        
        # Parse LaTeX segments
        parsed_latex = {
            i: self.latex_parser.parse(latex)
            for i, latex in enumerate(latex_segments)
        }
        
        # Extract mathematical expressions (non-LaTeX)
        math_expressions = self._extract_math_expressions(nlp_text)
        
        # Restore LaTeX and combine results
        combined_text = self._restore_latex(nlp_text, parsed_latex, placeholders)
        
        # Extract variables from both NLP and LaTeX
        variables = self._extract_variables(nlp_info, parsed_latex, math_expressions)
        
        return {
            "original_text": text,
            "processed_text": combined_text,
            "nlp_info": nlp_info,
            "latex_segments": parsed_latex,
            "math_expressions": math_expressions,
            "variables": variables,
            "sentences": [sent.text for sent in doc.sents],
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }
    
    def _extract_latex_segments(self, text: str) -> List[str]:
        """
        Extract LaTeX segments from text.
        
        Args:
            text: The text to process
            
        Returns:
            List of LaTeX segments
        """
        # Look for LaTeX delimiters like $...$ or $$...$$
        pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'
        matches = re.finditer(pattern, text, re.DOTALL)
        return [m.group(1) or m.group(2) for m in matches]
    
    def _replace_latex(self, text: str, latex_segments: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Replace LaTeX segments with placeholders for NLP processing.
        
        Args:
            text: The original text
            latex_segments: List of LaTeX segments
            
        Returns:
            Tuple of (text with placeholders, placeholder to LaTeX mapping)
        """
        modified_text = text
        placeholders = {}
        
        # Replace each LaTeX segment with a placeholder
        for i, latex in enumerate(latex_segments):
            placeholder = f"__LATEX_{i}__"
            # Escape regex special characters in latex
            latex_pattern = re.escape(f"${latex}$")
            modified_text = re.sub(latex_pattern, placeholder, modified_text)
            # Also try with double dollars
            latex_pattern = re.escape(f"$${latex}$$")
            modified_text = re.sub(latex_pattern, placeholder, modified_text)
            placeholders[placeholder] = latex
        
        return modified_text, placeholders
    
    def _extract_nlp_info(self, doc) -> Dict[str, Any]:
        """
        Extract basic NLP information from a spaCy document.
        
        Args:
            doc: The spaCy document
            
        Returns:
            Dictionary with NLP information
        """
        info = {
            "tokens": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "is_math": self._is_math_token(token)
                }
                for token in doc
            ],
            "noun_chunks": [
                {
                    "text": chunk.text,
                    "root": chunk.root.text,
                    "deps": [token.dep_ for token in chunk]
                }
                for chunk in doc.noun_chunks
            ],
            "sentences": [
                {
                    "text": sent.text,
                    "start_token": sent.start,
                    "end_token": sent.end
                }
                for sent in doc.sents
            ]
        }
        
        return info
    
    def _is_math_token(self, token) -> bool:
        """
        Check if a token is likely part of a mathematical expression.
        
        Args:
            token: A spaCy token
            
        Returns:
            True if the token is likely mathematical, False otherwise
        """
        # Check for common mathematical symbols
        if token.text in "+-*/=<>()[]{}^":
            return True
        
        # Check for variables (single letters, especially if they're nouns or proper nouns)
        if (len(token.text) == 1 and token.text.isalpha() and 
            token.pos_ in ["NOUN", "PROPN"]):
            return True
        
        # Check for numbers
        if token.text.isdigit() or (token.text.replace('.', '', 1).isdigit() and '.' in token.text):
            return True
        
        # Check for mathematical words
        math_words = ["sum", "product", "integral", "derivative", "function", "equals", 
                      "plus", "minus", "times", "divided", "greater", "less", "equal"]
        if token.lemma_.lower() in math_words:
            return True
        
        return False
    
    def _extract_math_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical expressions from text (outside of LaTeX).
        
        Args:
            text: The text to process
            
        Returns:
            List of dictionaries with expression information
        """
        expressions = []
        
        # Pattern for basic arithmetic expressions
        arith_pattern = r'([a-zA-Z0-9]+)\s*([+\-*/=<>])\s*([a-zA-Z0-9]+)'
        for match in re.finditer(arith_pattern, text):
            left = match.group(1)
            op = match.group(2)
            right = match.group(3)
            
            expressions.append({
                "text": match.group(0),
                "left": left,
                "operator": op,
                "right": right,
                "type": "arithmetic"
            })
        
        # Pattern for function applications
        func_pattern = r'([a-zA-Z]+)\(([^)]+)\)'
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            args = match.group(2)
            
            expressions.append({
                "text": match.group(0),
                "function": func_name,
                "arguments": [arg.strip() for arg in args.split(',')],
                "type": "function"
            })
        
        # Pattern for variables with subscripts
        subscript_pattern = r'([a-zA-Z])_([a-zA-Z0-9]+)'
        for match in re.finditer(subscript_pattern, text):
            base = match.group(1)
            subscript = match.group(2)
            
            expressions.append({
                "text": match.group(0),
                "base": base,
                "subscript": subscript,
                "type": "subscript"
            })
        
        return expressions
    
    def _restore_latex(self, text: str, parsed_latex: Dict[int, Dict[str, Any]], 
                      placeholders: Dict[str, str]) -> str:
        """
        Restore LaTeX segments in the text with their parsed versions.
        
        Args:
            text: The text with placeholders
            parsed_latex: Dictionary mapping indices to parsed LaTeX
            placeholders: Dictionary mapping placeholders to LaTeX strings
            
        Returns:
            Text with parsed LaTeX restored
        """
        restored_text = text
        
        for placeholder, latex in placeholders.items():
            idx = int(placeholder.replace("__LATEX_", "").replace("__", ""))
            parsed_info = parsed_latex.get(idx, {})
            parsed_text = parsed_info.get("parsed_text", latex)
            
            restored_text = restored_text.replace(placeholder, f"[MATH: {parsed_text}]")
        
        return restored_text
    
    def _extract_variables(self, nlp_info: Dict[str, Any], 
                          parsed_latex: Dict[int, Dict[str, Any]],
                          math_expressions: List[Dict[str, Any]]) -> List[str]:
        """
        Extract variables from NLP information and LaTeX.
        
        Args:
            nlp_info: Dictionary with NLP information
            parsed_latex: Dictionary with parsed LaTeX
            math_expressions: List of mathematical expressions
            
        Returns:
            List of variable names
        """
        variables = set()
        
        # Extract variables from NLP tokens
        for token in nlp_info['tokens']:
            if token['is_math'] and len(token['text']) == 1 and token['text'].isalpha():
                variables.add(token['text'])
        
        # Extract variables from LaTeX
        for latex_info in parsed_latex.values():
            variables.update(latex_info.get('variables', []))
        
        # Extract variables from math expressions
        for expr in math_expressions:
            if expr['type'] == 'arithmetic':
                left, right = expr['left'], expr['right']
                if len(left) == 1 and left.isalpha():
                    variables.add(left)
                if len(right) == 1 and right.isalpha():
                    variables.add(right)
            elif expr['type'] == 'subscript':
                variables.add(expr['base'])
            elif expr['type'] == 'function':
                for arg in expr['arguments']:
                    if len(arg) == 1 and arg.isalpha():
                        variables.add(arg)
        
        return sorted(list(variables))

def parse_mathematical_text(text: str) -> Dict[str, Any]:
    """
    Parse a mathematical text with mixed language and formulas.
    
    Args:
        text: The text to parse
        
    Returns:
        Dictionary with parsed information
    """
    parser = MathLanguageParser()
    return parser.parse(text)

def extract_math_elements(text: str) -> Dict[str, Any]:
    """
    Extract mathematical elements from text.
    
    Args:
        text: The text to process
        
    Returns:
        Dictionary with extracted elements
    """
    parser = MathLanguageParser()
    parse_result = parser.parse(text)
    
    # Extract key mathematical elements
    return {
        "variables": parse_result["variables"],
        "expressions": [
            expr["text"] for expr in parse_result["math_expressions"]
        ] + [
            latex_info["original"] for latex_info in parse_result["latex_segments"].values()
        ],
        "latex_segments": parse_result["latex_segments"],
        "sentences": parse_result["sentences"]
    }