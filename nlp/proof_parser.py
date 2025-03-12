"""
Proof parser for mathematical proofs.
Parses natural language proofs into structured representations with enhanced mathematical parsing.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Union, Optional

# Configure logging
logger = logging.getLogger("proof_parser")

# Import the enhanced mathematical language parser
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

class ProofParser:
    """
    Enhanced parser for mathematical proofs.
    Uses advanced mathematical language parsing for better understanding of expressions.
    """
    
    def __init__(self, use_advanced_parser: bool = True, kb=None):
        """
        Initialize the proof parser.
        
        Args:
            use_advanced_parser: Whether to use the advanced mathematical parser
            kb: Optional knowledge base
        """
        self.use_advanced_parser = use_advanced_parser
        self.kb = kb
        
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
        Parse a mathematical proof with enhanced mathematical expression handling.
        
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
        # If no specific theorem and proof were extracted, use the entire text
        if not theorem_text and not proof_text:
            theorem_text = proof_text = original_text
        
        # Parse theorem with the enhanced math parser
        if theorem_text:
            theorem_parse = self.math_parser.parse(theorem_text)
        else:
            theorem_parse = {}
        
        # Parse proof with the enhanced math parser
        if proof_text:
            proof_parse = self.math_parser.parse(proof_text)
        else:
            proof_parse = {}
        
        # Extract mathematical elements from both theorem and proof
        math_elements = {
            "theorem": extract_math_elements(theorem_text) if theorem_text else {},
            "proof": extract_math_elements(proof_text) if proof_text else {}
        }
        
        # Extract variables from math elements
        variables = self._extract_variables_from_math_elements(math_elements)
        
        # Extract mathematical expressions
        expressions = self._extract_expressions_from_math_elements(math_elements)
        
        # Extract statements from the proof
        parsed_statements = self._extract_statements_with_math(proof_parse)
        
        # Build the enhanced proof structure
        proof_structure = self._build_advanced_structure(parsed_statements, proof_parse, variables, expressions)
        
        return {
            "theorem_text": theorem_text,
            "proof_text": proof_text,
            "parsed_statements": parsed_statements,
            "proof_structure": proof_structure,
            "original_text": original_text,
            "variables": variables,
            "expressions": expressions,
            "theorem_parse": theorem_parse,
            "proof_parse": proof_parse,
            "math_elements": math_elements
        }
    
    def _extract_variables_from_math_elements(self, math_elements: Dict[str, Any]) -> List[str]:
        """
        Extract variables from math elements.
        
        Args:
            math_elements: Dictionary with math elements
            
        Returns:
            List of variables
        """
        variables = set()
        
        # Extract from theorem
        theorem_vars = math_elements.get("theorem", {}).get("variables", [])
        variables.update(theorem_vars)
        
        # Extract from proof
        proof_vars = math_elements.get("proof", {}).get("variables", [])
        variables.update(proof_vars)
        
        return sorted(list(variables))
    
    def _extract_expressions_from_math_elements(self, math_elements: Dict[str, Any]) -> List[str]:
        """
        Extract mathematical expressions from math elements.
        
        Args:
            math_elements: Dictionary with math elements
            
        Returns:
            List of expressions
        """
        expressions = set()
        
        # Extract from theorem
        theorem_exprs = math_elements.get("theorem", {}).get("expressions", [])
        expressions.update(theorem_exprs)
        
        # Extract from proof
        proof_exprs = math_elements.get("proof", {}).get("expressions", [])
        expressions.update(proof_exprs)
        
        return sorted(list(expressions))
    
    def _extract_statements_with_math(self, proof_parse: Dict[str, Any]) -> List[List[Tuple[str, str, str]]]:
        """
        Extract statements with mathematical annotations.
        
        Args:
            proof_parse: The parsed proof
            
        Returns:
            List of statements, each with token info
        """
        statements = []
        
        # Get sentences from the parse result
        sentences = proof_parse.get("sentences", [])
        
        for sentence in sentences:
            # Get tokens for this sentence
            sentence_tokens = []
            
            # Try to get tokens from NLP info
            if "nlp_info" in proof_parse and "tokens" in proof_parse["nlp_info"]:
                tokens = proof_parse["nlp_info"]["tokens"]
                
                # Find tokens for this sentence
                for token in tokens:
                    # Add token with available information
                    sentence_tokens.append((
                        token.get("text", ""),
                        token.get("pos", ""),
                        token.get("dep", "")
                    ))
            else:
                # Fallback: split sentence into tokens
                words = re.findall(r'\b\w+\b|[^\w\s]', sentence)
                sentence_tokens = [(word, "", "") for word in words]
            
            # Add mathematical annotations
            sentence_tokens = self._annotate_math_tokens(sentence_tokens, proof_parse)
            
            # Only add non-empty statements
            if sentence_tokens:
                statements.append(sentence_tokens)
        
        return statements
    
    def _annotate_math_tokens(self, tokens: List[Tuple[str, str, str]], proof_parse: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Annotate tokens with mathematical information.
        
        Args:
            tokens: List of (token, pos, dep) tuples
            proof_parse: The parsed proof
            
        Returns:
            Annotated tokens
        """
        # Get mathematical expressions
        math_expressions = proof_parse.get("math_expressions", [])
        
        # Simple annotation: mark tokens that appear in mathematical expressions
        annotated_tokens = []
        for token, pos, dep in tokens:
            is_math = False
            
            # Check if token appears in any mathematical expression
            for expr in math_expressions:
                if token in expr.get("text", ""):
                    is_math = True
                    break
            
            # Check if token is a variable
            if len(token) == 1 and token.isalpha() and token in proof_parse.get("variables", []):
                is_math = True
            
            # Add math annotation to POS tag if it's mathematical
            if is_math:
                pos = f"MATH_{pos}" if pos else "MATH"
            
            annotated_tokens.append((token, pos, dep))
        
        return annotated_tokens
    
    def _build_advanced_structure(self, parsed_statements: List[List[Tuple[str, str, str]]], 
                                  proof_parse: Dict[str, Any],
                                  variables: List[str],
                                  expressions: List[str]) -> Dict[str, Any]:
        """
        Build an enhanced proof structure with mathematical elements.
        
        Args:
            parsed_statements: The parsed statements
            proof_parse: The parsed proof
            variables: List of variables
            expressions: List of expressions
            
        Returns:
            Dictionary with proof structure information
        """
        structure = {
            "assumptions": [],
            "conclusions": [],
            "proof_methods": [],
            "variables": variables,
            "expressions": expressions,
            "mathematical_elements": []
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
        
        # Add mathematical elements
        for expr in expressions:
            # Try to determine what kind of expression this is
            expr_type = self._classify_expression(expr, proof_parse)
            
            structure["mathematical_elements"].append({
                "expression": expr,
                "type": expr_type,
                "variables": [var for var in variables if var in expr]
            })
        
        # Integrate latex segments if available
        for latex_info in proof_parse.get("latex_segments", {}).values():
            if "parsed_text" in latex_info:
                structure["mathematical_elements"].append({
                    "expression": latex_info.get("original", ""),
                    "parsed": latex_info.get("parsed_text", ""),
                    "type": latex_info.get("expression_type", "unknown"),
                    "variables": latex_info.get("variables", [])
                })
        
        # Check for evenness proof (common pattern)
        structure["is_evenness_proof"] = self._is_evenness_proof(sentences, variables, expressions)
        
        return structure
    
    def _parse_with_basic_parser(self, theorem_text: str, proof_text: str, original_text: str) -> Dict[str, Any]:
        """
        Parse using the basic parser (fallback method).
        
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
    
    def _build_basic_structure(self, parsed_statements: List[List[Tuple[str, str, str]]], variables: List[str]) -> Dict[str, Any]:
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
    
    def _classify_expression(self, expr: str, proof_parse: Dict[str, Any]) -> str:
        """
        Classify a mathematical expression.
        
        Args:
            expr: The expression text
            proof_parse: The parsed proof
            
        Returns:
            Expression type
        """
        # Check if it contains an equals sign (equation)
        if "=" in expr:
            return "equation"
        
        # Check if it contains comparison operators
        if any(op in expr for op in ["<", ">", "≤", "≥", "≠"]):
            return "inequality"
        
        # Check if it contains arithmetic operators
        if any(op in expr for op in ["+", "-", "*", "/", "^"]):
            return "arithmetic"
        
        # Check if it looks like a function application
        if re.search(r'[a-zA-Z]\([^)]+\)', expr):
            return "function"
        
        # Default to expression
        return "expression"
    
    def _is_evenness_proof(self, sentences: List[str], variables: List[str], expressions: List[str]) -> bool:
        """
        Detect if this is an evenness proof.
        
        Args:
            sentences: List of sentences
            variables: List of variables
            expressions: List of expressions
            
        Returns:
            True if this is an evenness proof, False otherwise
        """
        # Join sentences for easier analysis
        text = " ".join(sentences).lower()
        
        # Check for key evenness indicators
        if "even" not in text:
            return False
        
        # Check for expressions of the form x+x or 2*x
        for var in variables:
            if f"{var}+{var}" in "".join(expressions) or f"{var} + {var}" in text:
                return True
            if f"2*{var}" in "".join(expressions) or f"2 * {var}" in text:
                return True
        
        # Check for statements about divisibility by 2
        if "divisible by 2" in text or "multiple of 2" in text:
            return True
        
        # Check for explicit mention of evenness
        for var in variables:
            if f"{var} is even" in text:
                return True
        
        return False


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
    
    # Check for theorem marker
    for i, line in enumerate(lines):
        if "theorem" in line.lower() or "lemma" in line.lower():
            if i < len(lines) - 1:
                theorem = line.strip()
                proof = "\n".join(lines[i+1:]).strip()
                return theorem, proof
    
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

def parse_math_proof(text: str, kb=None) -> Dict[str, Any]:
    """
    Parse a mathematical proof and return structured information.
    Uses enhanced mathematical language parsing.
    
    Args:
        text: The input text containing theorem and proof
        kb: Optional knowledge base
        
    Returns:
        Dictionary with parsed information
    """
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Create parser and parse the proof
    parser = ProofParser(kb=kb)
    parsed_info = parser.parse(preprocessed_text)
    
    return parsed_info