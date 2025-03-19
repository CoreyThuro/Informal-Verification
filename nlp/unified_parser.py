"""
Unified parser for mathematical proofs and formulas.
Combines proof structure parsing and mathematical language parsing.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import os

# Configure logging
logger = logging.getLogger("unified_parser")

# Try to import spaCy
try:
    import spacy
    HAS_SPACY = True
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        logger.warning("spaCy model 'en_core_web_sm' not found. Using simplified parsing.")
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Using simplified parsing.")
    HAS_SPACY = False
    nlp = None

# Import existing parsing components
from nlp.latex_parser import LatexParser

class UnifiedProofParser:
    """
    Unified parser that handles both proof structure and mathematical language.
    Combines functionality from proof_parser.py and math_language_parser.py.
    """
    
    def __init__(self, use_latex_parser: bool = True, kb=None):
        """
        Initialize the unified parser.
        
        Args:
            use_latex_parser: Whether to use the LaTeX parser for mathematical expressions
            kb: Optional knowledge base for domain-specific parsing
        """
        self.use_latex_parser = use_latex_parser
        self.kb = kb
        
        # Initialize the LaTeX parser if requested
        if use_latex_parser:
            try:
                from nlp.latex_parser import LatexParser
                self.latex_parser = LatexParser()
                logger.info("Initialized LaTeX parser")
            except Exception as e:
                logger.warning(f"Failed to initialize LaTeX parser: {e}")
                self.latex_parser = None
                self.use_latex_parser = False
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a mathematical proof with integrated formula handling.
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary with parsed information
        """
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        # Split into theorem and proof
        theorem_text, proof_text = self._split_theorem_and_proof(preprocessed_text)
        
        # First, extract and parse mathematical expressions
        math_segments = self._extract_and_parse_math(preprocessed_text)
        
        # Next, parse the natural language structure
        statements = self._extract_statements(preprocessed_text, math_segments)
        
        # Build proof structure
        structure = self._build_structure(statements, math_segments, theorem_text, proof_text)
        
        # Extract variables from all sources
        variables = self._extract_variables(statements, math_segments, theorem_text, proof_text)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "parsed_statements": statements,
            "proof_structure": structure,
            "math_segments": math_segments,
            "variables": variables,
            "original_text": text
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text to normalize mathematical notation.
        
        Args:
            text: The input text
            
        Returns:
            Normalized text
        """
        # Replace common mathematical symbols
        replacements = {
            '∀': 'for all',
            '∃': 'there exists',
            '∈': 'in',
            '⊆': 'subset of',
            '⊂': 'proper subset of',
            '∩': 'intersection',
            '∪': 'union',
            '⇒': 'implies',
            '→': 'implies',
            '⟹': 'implies',
            '⟺': 'if and only if',
            '⟷': 'if and only if',
            '≠': 'not equal to',
            '≤': 'less than or equal to',
            '≥': 'greater than or equal to',
            '≡': 'equivalent to',
            '≈': 'approximately equal to',
            '∞': 'infinity',
            '√': 'square root of',
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral',
            '∂': 'partial'
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, f' {replacement} ')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _split_theorem_and_proof(self, text: str) -> Tuple[str, str]:
        """
        Split the input text into theorem statement and proof.
        
        Args:
            text: The input text containing theorem and proof
            
        Returns:
            Tuple of (theorem_statement, proof_text)
        """
        # Look for "Proof:" or similar markers
        proof_markers = [
            r'Proof[\s\:]+',
            r'Proof[\s\.]+'
        ]
        
        for marker in proof_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                split_index = match.start()
                theorem = text[:split_index].strip()
                proof = text[match.end():].strip()
                return theorem, proof
        
        # No explicit marker found, try to infer
        lines = text.split("\n")
        
        # If short, single line, it's likely just a theorem
        if len(lines) <= 1:
            return text, ""
        
        # Otherwise, first line is often the theorem
        theorem = lines[0].strip()
        proof = "\n".join(lines[1:]).strip()
        
        return theorem, proof
    
    def _extract_and_parse_math(self, text: str) -> Dict[str, Any]:
        """
        Extract and parse mathematical expressions from the text.
        
        Args:
            text: The text to process
            
        Returns:
            Dictionary with parsed math segments
        """
        math_segments = {}
        
        # Extract LaTeX segments if the parser is available
        if self.use_latex_parser and hasattr(self, 'latex_parser') and self.latex_parser:
            latex_segments = self._extract_latex_segments(text)
            
            # Parse each LaTeX segment
            for i, latex in enumerate(latex_segments):
                try:
                    parsed = self.latex_parser.parse(latex)
                    math_segments[f"latex_{i}"] = {
                        "original": latex,
                        "parsed": parsed,
                        "type": "latex"
                    }
                except Exception as e:
                    logger.warning(f"Error parsing LaTeX segment: {e}")
                    math_segments[f"latex_{i}"] = {
                        "original": latex,
                        "parsed": None,
                        "type": "latex",
                        "error": str(e)
                    }
        
        # Extract other mathematical expressions
        other_math = self._extract_math_expressions(text)
        for i, expr in enumerate(other_math):
            math_segments[f"expr_{i}"] = {
                "original": expr["text"],
                "parsed": expr,
                "type": "expression"
            }
        
        return math_segments
    
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
    
    def _extract_statements(self, text: str, math_segments: Dict[str, Any]) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements from the text.
        
        Args:
            text: The text to process
            math_segments: Dictionary of parsed math segments
            
        Returns:
            List of statements, each represented as a list of (token, pos, dep) tuples
        """
        # Replace math segments with placeholders to avoid parsing issues
        processed_text, placeholders = self._replace_math_with_placeholders(text, math_segments)
        
        # Use spaCy if available, otherwise fall back to regex
        if HAS_SPACY and nlp:
            statements = self._extract_statements_spacy(processed_text)
        else:
            statements = self._extract_statements_regex(processed_text)
        
        # Restore math segments in statements
        statements = self._restore_math_in_statements(statements, placeholders, math_segments)
        
        return statements
    
    def _replace_math_with_placeholders(self, text: str, math_segments: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """
        Replace math segments with placeholders for NLP processing.
        
        Args:
            text: The original text
            math_segments: Dictionary of parsed math segments
            
        Returns:
            Tuple of (text with placeholders, placeholder to segment key mapping)
        """
        modified_text = text
        placeholders = {}
        
        # Replace LaTeX segments
        for key, segment in math_segments.items():
            if segment["type"] == "latex":
                original = segment["original"]
                placeholder = f"__MATH_{key}__"
                
                # Replace the LaTeX in the text
                modified_text = modified_text.replace(f"${original}$", placeholder)
                modified_text = modified_text.replace(f"$${original}$$", placeholder)
                
                placeholders[placeholder] = key
            
        # Replace other math expressions
        for key, segment in math_segments.items():
            if segment["type"] == "expression":
                original = segment["original"]
                if original in modified_text:  # Only replace if it still exists
                    placeholder = f"__MATH_{key}__"
                    modified_text = modified_text.replace(original, placeholder)
                    placeholders[placeholder] = key
        
        return modified_text, placeholders
    
    def _extract_statements_spacy(self, text: str) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements using spaCy.
        
        Args:
            text: The text to process
            
        Returns:
            List of statements, each represented as a list of (token, pos, dep) tuples
        """
        statements = []
        
        # Parse the text with spaCy
        doc = nlp(text)
        
        # Extract sentences
        for sent in doc.sents:
            sentence_tokens = [(token.text, token.pos_, token.dep_) for token in sent]
            statements.append(sentence_tokens)
            
        return statements
    
    def _extract_statements_regex(self, text: str) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements using regex (fallback method).
        
        Args:
            text: The text to process
            
        Returns:
            List of statements, each represented as a list of (token, pos, dep) tuples
        """
        statements = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # Split into tokens
            tokens = re.findall(r'\b\w+\b|[^\w\s]', sentence.strip())
            
            # Create simple token tuples (without POS or dependency info)
            sentence_tokens = [(token, "", "") for token in tokens]
            
            if sentence_tokens:
                statements.append(sentence_tokens)
        
        return statements
    
    def _restore_math_in_statements(self, statements: List[List[Tuple[str, str, str]]], 
                                  placeholders: Dict[str, str], 
                                  math_segments: Dict[str, Any]) -> List[List[Tuple[str, str, str]]]:
        """
        Restore math segments in statements.
        
        Args:
            statements: List of statements with placeholders
            placeholders: Mapping from placeholders to segment keys
            math_segments: Dictionary of parsed math segments
            
        Returns:
            List of statements with math segments restored
        """
        restored_statements = []
        
        for statement in statements:
            restored_statement = []
            
            for token, pos, dep in statement:
                if token in placeholders:
                    # Replace placeholder with math segment
                    math_key = placeholders[token]
                    math_segment = math_segments[math_key]
                    
                    # Use original text and mark as a math token
                    restored_statement.append((math_segment["original"], "MATH", "math"))
                else:
                    restored_statement.append((token, pos, dep))
            
            restored_statements.append(restored_statement)
        
        return restored_statements
    
    def _build_structure(self, statements: List[List[Tuple[str, str, str]]], 
                        math_segments: Dict[str, Any],
                        theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Build the proof structure from parsed statements.
        
        Args:
            statements: The parsed statements
            math_segments: Dictionary of parsed math segments
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Dictionary with proof structure information
        """
        structure = {
            "assumptions": [],
            "conclusions": [],
            "proof_methods": [],
            "variables": [],  # This will be filled later
            "expressions": []
        }
        
        # Get sentences
        sentences = [' '.join([token[0] for token in stmt]) for stmt in statements]
        
        # Identify assumptions, conclusions, and proof methods
        assumption_markers = ["assume", "let", "suppose", "given", "if", "for any", "for all"]
        conclusion_markers = ["therefore", "thus", "hence", "so", "we have", "we get", "conclude"]
        proof_method_markers = {
            "induction": ["induction", "base case", "inductive", "hypothesis"],
            "contradiction": ["contradiction", "contrary", "false", "absurd", "suppose not"],
            "case": ["case", "cases", "first case", "second case"]
        }
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for assumptions
            for marker in assumption_markers:
                if marker in sentence_lower:
                    structure["assumptions"].append((sentence, "intros"))
                    break
            
            # Check for conclusions
            for marker in conclusion_markers:
                if marker in sentence_lower:
                    structure["conclusions"].append((sentence, "assert"))
                    break
            
            # Check for proof methods
            for method, markers in proof_method_markers.items():
                for marker in markers:
                    if marker in sentence_lower:
                        structure["proof_methods"].append((method, method, sentence))
                        break
        
        # Extract expressions from math segments
        for key, segment in math_segments.items():
            structure["expressions"].append(segment["original"])
        
        return structure
    
    def _extract_variables(self, statements: List[List[Tuple[str, str, str]]], 
                          math_segments: Dict[str, Any],
                          theorem_text: str, proof_text: str) -> List[str]:
        """
        Extract variables from statements and math segments.
        
        Args:
            statements: The parsed statements
            math_segments: Dictionary of parsed math segments
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            List of variables
        """
        variables = set()
        
        # Extract from statements (single letters)
        for statement in statements:
            for token, pos, dep in statement:
                if len(token) == 1 and token.isalpha():
                    variables.add(token)
        
        # Extract from math segments
        for key, segment in math_segments.items():
            if segment["type"] == "latex" and segment["parsed"]:
                if "variables" in segment["parsed"]:
                    variables.update(segment["parsed"]["variables"])
            elif segment["type"] == "expression":
                parsed = segment["parsed"]
                if parsed["type"] == "arithmetic":
                    if len(parsed["left"]) == 1 and parsed["left"].isalpha():
                        variables.add(parsed["left"])
                    if len(parsed["right"]) == 1 and parsed["right"].isalpha():
                        variables.add(parsed["right"])
                elif parsed["type"] == "subscript":
                    variables.add(parsed["base"])
                elif parsed["type"] == "function":
                    for arg in parsed["arguments"]:
                        if len(arg) == 1 and arg.isalpha():
                            variables.add(arg)
        
        # Sort variables for consistency
        return sorted(list(variables))


def parse_mathematical_proof(text: str, kb=None) -> Dict[str, Any]:
    """
    Parse a mathematical proof using the unified parser.
    
    Args:
        text: The text to parse
        kb: Optional knowledge base
        
    Returns:
        Dictionary with parsed information
    """
    parser = UnifiedProofParser(kb=kb)
    return parser.parse(text)