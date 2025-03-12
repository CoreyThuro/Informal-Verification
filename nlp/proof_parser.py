"""
Proof parser for mathematical proofs.
Parses natural language proofs into structured representations.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Union, Optional

# Configure logging
logger = logging.getLogger("proof_parser")

# Import the mathematical language parser
from nlp.math_language_parser import MathLanguageParser, extract_math_elements

# Try to load spaCy
try:
    import spacy
    HAS_SPACY = True
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        logger.warning("spaCy model 'en_core_web_sm' not found. Please install with 'python -m spacy download en_core_web_sm'")
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Using simplified parsing.")
    HAS_SPACY = False
    nlp = None

# Try to import SymPy for formula parsing
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, TokenError
    HAS_SYMPY = True
except ImportError:
    logger.warning("SymPy not available. Formula parsing will be limited.")
    HAS_SYMPY = False

class ProofParser:
    """
    Parser for mathematical proofs.
    """
    
    def __init__(self, use_advanced_parser: bool = True):
        """
        Initialize the proof parser.
        
        Args:
            use_advanced_parser: Whether to use the advanced mathematical parser
        """
        self.use_advanced_parser = use_advanced_parser
        
        # Initialize the mathematical language parser if requested
        if use_advanced_parser:
            try:
                self.math_parser = MathLanguageParser()
                logger.info("Using advanced mathematical language parser")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced parser: {e}. Falling back to basic parser.")
                self.math_parser = None
                self.use_advanced_parser = False
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a mathematical proof.
        
        Args:
            text: The proof text
            
        Returns:
            Dictionary with parsed information
        """
        # Split into theorem and proof
        theorem_text, proof_text = split_theorem_and_proof(text)
        
        # Parse with the appropriate parser
        if self.use_advanced_parser and hasattr(self, 'math_parser') and self.math_parser:
            return self._parse_with_advanced_parser(theorem_text, proof_text, text)
        else:
            return self._parse_with_basic_parser(theorem_text, proof_text, text)
    
    def _parse_with_advanced_parser(self, theorem_text: str, proof_text: str, original_text: str) -> Dict[str, Any]:
        """
        Parse using the advanced mathematical language parser.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            original_text: The original input text
            
        Returns:
            Dictionary with parsed information
        """
        # Parse theorem
        if theorem_text:
            theorem_parse = self.math_parser.parse(theorem_text)
        else:
            theorem_parse = {}
        
        # Parse proof
        if proof_text:
            proof_parse = self.math_parser.parse(proof_text)
        else:
            proof_parse = self.math_parser.parse(original_text)
            # If no specific theorem and proof were extracted, use the entire text
            if not theorem_text and not proof_text:
                theorem_text = proof_text = original_text
        
        # Extract structured information
        variables = list(set(
            theorem_parse.get("variables", []) + 
            proof_parse.get("variables", [])
        ))
        
        # Extract statements from the proof
        parsed_statements = self._extract_statements(proof_parse)
        
        # Build the proof structure
        proof_structure = self._build_structure(parsed_statements, proof_parse, variables)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "parsed_statements": parsed_statements,
            "proof_structure": proof_structure,
            "original_text": original_text,
            "variables": variables,
            "theorem_parse": theorem_parse,
            "proof_parse": proof_parse
        }
    
    def _parse_with_basic_parser(self, theorem_text: str, proof_text: str, original_text: str) -> Dict[str, Any]:
        """
        Parse using the basic parser.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            original_text: The original input text
            
        Returns:
            Dictionary with parsed information
        """
        # If no specific theorem and proof were extracted, use the entire text
        if not theorem_text and not proof_text:
            theorem_text = proof_text = original_text
        
        # Parse theorem and proof with spaCy if available
        if HAS_SPACY and nlp:
            theorem_doc = nlp(theorem_text) if theorem_text else None
            proof_doc = nlp(proof_text) if proof_text else None
            
            # Extract statements from the proof
            parsed_statements = self._extract_statements_spacy(proof_doc) if proof_doc else []
            
            # Extract variables
            variables = self._extract_variables(theorem_doc, proof_doc)
        else:
            # Fallback to simpler parsing
            parsed_statements = self._extract_statements_regex(proof_text)
            variables = self._extract_variables_regex(theorem_text, proof_text)
        
        # Build the proof structure
        proof_structure = self._build_basic_structure(parsed_statements, variables)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "parsed_statements": parsed_statements,
            "proof_structure": proof_structure,
            "original_text": original_text,
            "variables": variables
        }
    
    def _extract_statements(self, proof_parse: Dict[str, Any]) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements from a parsed proof.
        
        Args:
            proof_parse: The parsed proof data
            
        Returns:
            List of statements, each represented as a list of (token, pos, dep) tuples
        """
        statements = []
        
        # Get sentences from the parse result
        sentences = proof_parse.get("sentences", [])
        
        for sentence in sentences:
            # Get tokens for this sentence
            sentence_tokens = []
            for token in proof_parse.get("nlp_info", {}).get("tokens", []):
                if "sentences" in proof_parse.get("nlp_info", {}):
                    # Check if token is in this sentence (using token indices)
                    for sent_info in proof_parse["nlp_info"]["sentences"]:
                        if (
                            token.get("index", -1) >= sent_info.get("start_token", 0) and
                            token.get("index", -1) < sent_info.get("end_token", 0)
                        ):
                            sentence_tokens.append((
                                token.get("text", ""),
                                token.get("pos", ""),
                                token.get("dep", "")
                            ))
                else:
                    # If sentence boundaries aren't available, just collect all tokens
                    sentence_tokens.append((
                        token.get("text", ""),
                        token.get("pos", ""),
                        token.get("dep", "")
                    ))
            
            # Only add non-empty statements
            if sentence_tokens:
                statements.append(sentence_tokens)
            
        return statements
    
    def _extract_statements_spacy(self, doc) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements from a spaCy document.
        
        Args:
            doc: The spaCy document
            
        Returns:
            List of statements, each represented as a list of (token, pos, dep) tuples
        """
        statements = []
        
        # Extract sentences
        for sent in doc.sents:
            sentence_tokens = [(token.text, token.pos_, token.dep_) for token in sent]
            statements.append(sentence_tokens)
            
        return statements
    
    def _extract_statements_regex(self, text: str) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements using regex (fallback method).
        
        Args:
            text: The text to parse
            
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
    
    def _extract_variables(self, theorem_doc, proof_doc) -> List[str]:
        """
        Extract variables from spaCy documents.
        
        Args:
            theorem_doc: The theorem spaCy document
            proof_doc: The proof spaCy document
            
        Returns:
            List of variables
        """
        variables = set()
        
        # Process theorem
        if theorem_doc:
            for token in theorem_doc:
                # Check for potential variables (single letters)
                if len(token.text) == 1 and token.text.isalpha():
                    variables.add(token.text)
        
        # Process proof
        if proof_doc:
            for token in proof_doc:
                # Check for potential variables (single letters)
                if len(token.text) == 1 and token.text.isalpha():
                    variables.add(token.text)
        
        return sorted(list(variables))
    
    def _extract_variables_regex(self, theorem_text: str, proof_text: str) -> List[str]:
        """
        Extract variables using regex (fallback method).
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            List of variables
        """
        variables = set()
        
        # Find single-letter tokens that might be variables
        for text in [theorem_text, proof_text]:
            if text:
                # Find single letters that are likely variables
                var_candidates = re.findall(r'\b([a-zA-Z])\b', text)
                variables.update(var_candidates)
        
        return sorted(list(variables))
    
    def _build_structure(self, 
                        parsed_statements: List[List[Tuple[str, str, str]]], 
                        proof_parse: Dict[str, Any],
                        variables: List[str]) -> Dict[str, Any]:
        """
        Build the proof structure from parsed statements.
        
        Args:
            parsed_statements: The parsed statements
            proof_parse: The parsed proof data
            variables: The variables in the proof
            
        Returns:
            Dictionary with proof structure information
        """
        structure = {
            "assumptions": [],
            "conclusions": [],
            "proof_methods": [],
            "variables": variables,
            "expressions": []
        }
        
        # Get sentences
        sentences = [' '.join([token[0] for token in stmt]) for stmt in parsed_statements]
        
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
        
        # Extract expressions from the proof parse
        for expr in proof_parse.get("math_expressions", []):
            structure["expressions"].append(expr["text"])
        
        # Add expressions from LaTeX segments
        for latex_info in proof_parse.get("latex_segments", {}).values():
            if "original" in latex_info:
                structure["expressions"].append(latex_info["original"])
        
        return structure
    
    def _build_basic_structure(self, 
                              parsed_statements: List[List[Tuple[str, str, str]]], 
                              variables: List[str]) -> Dict[str, Any]:
        """
        Build a basic proof structure (fallback method).
        
        Args:
            parsed_statements: The parsed statements
            variables: The variables in the proof
            
        Returns:
            Dictionary with proof structure information
        """
        structure = {
            "assumptions": [],
            "conclusions": [],
            "proof_methods": [],
            "variables": variables,
            "expressions": []
        }
        
        # Get sentences
        sentences = [' '.join([token[0] for token in stmt]) for stmt in parsed_statements]
        
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
        
        # Extract expressions
        for sentence in sentences:
            # Simple regex to find mathematical expressions
            expr_pattern = r'\b([a-zA-Z])(?:\s*[\+\-\*\/\^]\s*([a-zA-Z0-9]+))+\b'
            expressions = re.findall(expr_pattern, sentence)
            if expressions:
                for expr in expressions:
                    if isinstance(expr, tuple):
                        structure["expressions"].append(''.join(expr))
                    else:
                        structure["expressions"].append(expr)
        
        return structure

def split_theorem_and_proof(text: str) -> Tuple[str, str]:
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

def preprocess_text(text: str) -> str:
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

def parse_math_proof(text: str) -> Dict[str, Any]:
    """
    Parse a mathematical proof and return structured information.
    
    Args:
        text: The input text containing theorem and proof
        
    Returns:
        Dictionary with parsed information
    """
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Create parser and parse the proof
    parser = ProofParser()
    parsed_info = parser.parse(preprocessed_text)
    
    return parsed_info