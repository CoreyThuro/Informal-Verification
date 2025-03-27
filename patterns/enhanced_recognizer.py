"""
Enhanced pattern recognition for mathematical proofs.
Uses NLP techniques and proof structure modeling for deeper semantic understanding.
"""

import re
import spacy
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter

# Import the NLP analyzer module
from patterns.nlp_analyzer import (
    analyze_proof, 
    extract_math_entities, 
    score_patterns_nlp, 
    extract_variables,
    identify_proof_steps,
    get_enhanced_pattern
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback to a simpler model if the full model isn't available
    nlp = spacy.blank("en")

class ProofStructureAnalyzer:
    """
    Analyzes the structure of mathematical proofs using NLP techniques.
    """
    
    def __init__(self):
        # Proof step markers
        self.step_markers = {
            "assumption": ["assume", "suppose", "let", "consider", "given"],
            "deduction": ["therefore", "thus", "hence", "so", "consequently", "it follows that"],
            "case_intro": ["case", "if", "when", "consider the case"],
            "base_case": ["base case", "when n = 0", "for n = 0", "initial case"],
            "inductive_step": ["inductive step", "inductive case", "induction hypothesis", "assume for k"],
            "contradiction": ["contradiction", "absurd", "impossible", "this contradicts"]
        }
        
        # Mathematical concept terms
        self.concept_terms = {
            "number_theory": ["prime", "divisible", "factor", "gcd", "lcm", "modulo", "congruent", "even", "odd"],
            "algebra": ["group", "ring", "field", "vector", "matrix", "polynomial", "linear", "commutative"],
            "analysis": ["limit", "continuous", "derivative", "integral", "sequence", "series", "convergent"],
            "topology": ["open", "closed", "compact", "connected", "neighborhood", "homeomorphic"]
        }
        
        # Common mathematical relations
        self.relations = ["equals", "less than", "greater than", "divides", "contains", "subset", "element of"]

    def segment_proof(self, proof_text: str) -> List[Dict[str, Any]]:
        """
        Segment a proof into logical steps.
        
        Args:
            proof_text: The proof text
            
        Returns:
            List of proof steps with their types and content
        """
        # Normalize whitespace
        proof_text = re.sub(r'\s+', ' ', proof_text).strip()
        
        # Split into sentences
        doc = nlp(proof_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        steps = []
        for sentence in sentences:
            step_type = self._classify_step(sentence)
            steps.append({
                "text": sentence,
                "type": step_type,
                "entities": self._extract_entities(sentence)
            })
            
        return steps
    
    def _classify_step(self, sentence: str) -> str:
        """Classify the type of proof step."""
        sentence_lower = sentence.lower()
        
        for step_type, markers in self.step_markers.items():
            for marker in markers:
                if marker in sentence_lower:
                    return step_type
        
        # Default to "statement" if no specific type is identified
        return "statement"
    
    def _extract_entities(self, sentence: str) -> List[Dict[str, str]]:
        """Extract mathematical entities from a sentence."""
        doc = nlp(sentence)
        entities = []
        
        # Extract variables (single letters)
        variables = set(re.findall(r'\b([a-zA-Z])\b', sentence))
        
        # Extract expressions (simple patterns)
        expressions = re.findall(r'([a-zA-Z0-9\s\+\-\*/\^=<>]+)', sentence)
        
        # Extract numbers
        numbers = [token.text for token in doc if token.like_num]
        
        # Combine entities
        for var in variables:
            entities.append({"type": "variable", "text": var})
        
        for expr in expressions:
            if any(op in expr for op in ['+', '-', '*', '/', '^', '=', '<', '>']):
                entities.append({"type": "expression", "text": expr.strip()})
        
        for num in numbers:
            entities.append({"type": "number", "text": num})
            
        return entities

    def detect_proof_pattern(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Detect the pattern of a proof using NLP and structure analysis.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            Dictionary with pattern info
        """
        # Segment the proof
        steps = self.segment_proof(proof_text)
        
        # Count step types
        step_types = Counter([step["type"] for step in steps])
        
        # Extract all entities
        all_entities = []
        for step in steps:
            all_entities.extend(step["entities"])
        
        # Analyze theorem statement
        theorem_doc = nlp(theorem_text)
        
        # Detect pattern based on steps and entities
        pattern_info = {
            "steps": steps,
            "variables": self._extract_variables(theorem_text, proof_text),
            "step_types": dict(step_types),
            "entities": all_entities
        }
        
        # Determine the pattern
        pattern = self._determine_pattern(pattern_info)
        pattern_info["pattern"] = pattern
        
        return pattern_info
    
    def _determine_pattern(self, pattern_info: Dict[str, Any]) -> str:
        """Determine the proof pattern based on analysis."""
        step_types = pattern_info["step_types"]
        
        # Check for induction pattern
        if step_types.get("base_case", 0) > 0 and step_types.get("inductive_step", 0) > 0:
            return "induction"
        
        # Check for contradiction pattern
        if step_types.get("contradiction", 0) > 0 or step_types.get("assumption", 0) > 0:
            # Check if we're assuming the opposite of what we want to prove
            return "contradiction"
        
        # Check for case analysis
        if step_types.get("case_intro", 0) > 1:
            return "cases"
        
        # Check for evenness (more specific than just keyword matching)
        if self._is_evenness_proof(pattern_info):
            return "evenness"
        
        # Default to direct proof
        return "direct"
    
    def _is_evenness_proof(self, pattern_info: Dict[str, Any]) -> bool:
        """Check if this is an evenness proof with deeper analysis."""
        # Look for evenness concepts in entities
        entities = pattern_info["entities"]
        
        # Check for expressions related to evenness
        expressions = [e["text"] for e in entities if e["type"] == "expression"]
        
        # Check for patterns like x+x, 2*x, or divisibility by 2
        has_x_plus_x = any(re.search(r'([a-z])\s*\+\s*\1', expr) for expr in expressions)
        has_2x = any(re.search(r'2\s*\*\s*[a-z]', expr) for expr in expressions)
        has_divisibility = any("divisible by 2" in expr or "even" in expr for expr in expressions)
        
        return has_x_plus_x or has_2x or has_divisibility
    
    def _extract_variables(self, theorem_text: str, proof_text: str) -> List[str]:
        """Extract variables with more sophisticated analysis."""
        combined = f"{theorem_text} {proof_text}"
        
        # Use spaCy to analyze the text
        doc = nlp(combined)
        
        # Look for single letter variables
        variables = set(re.findall(r'\b([a-z])\b', combined))
        
        # Look for variables in "let x be" or "for all x" patterns
        for token in doc:
            if token.text.lower() in ["let", "for", "all", "any", "some", "every"]:
                # Check the next few tokens for potential variables
                for i in range(1, 4):
                    if token.i + i < len(doc) and len(doc[token.i + i].text) == 1 and doc[token.i + i].text.isalpha():
                        variables.add(doc[token.i + i].text)
        
        return sorted(list(variables))

def enhanced_recognize_pattern(theorem_text: str, proof_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced pattern recognition for mathematical proofs using advanced NLP techniques.
    
    Args:
        theorem_text: The theorem statement
        proof_text: The proof text
        
    Returns:
        Tuple of (pattern_name, pattern_info)
    """
    # Perform a more direct pattern analysis first
    direct_pattern = _analyze_direct_patterns(theorem_text, proof_text)
    
    # Perform NLP-based analysis on the proof text
    nlp_doc = nlp(proof_text)
    
    # Extract linguistic features
    linguistic_features = {
        "pos_tags": Counter([token.pos_ for token in nlp_doc]),
        "dependencies": [(token.text, token.dep_, token.head.text) for token in nlp_doc if token.dep_ != "punct"],
        "entities": [(ent.text, ent.label_) for ent in nlp_doc.ents],
        "noun_chunks": [chunk.text for chunk in nlp_doc.noun_chunks]
    }
    
    # Enhance direct pattern with linguistic features
    for token in nlp_doc:
        # Look for induction indicators
        if token.lemma_ in ["induct", "base", "hypothesis", "step"]:
            direct_pattern["scores"]["induction"] += 1
            
        # Look for contradiction indicators
        if token.lemma_ in ["contradict", "contrary", "impossible", "absurd"]:
            direct_pattern["scores"]["contradiction"] += 1
            
        # Look for case analysis indicators
        if token.lemma_ in ["case", "consider"] or (token.text == "if" and any(t.text == "then" for t in token.children)):
            direct_pattern["scores"]["cases"] += 1
            
        # Look for evenness indicators
        if token.lemma_ in ["even", "odd", "divisible"] or (token.text == "2" and any(t.dep_ == "nummod" for t in token.children)):
            direct_pattern["scores"]["evenness"] += 1
    
    # Recalculate confidence based on enhanced scores
    total_score = sum(direct_pattern["scores"].values())
    if total_score > 0:
        max_pattern = max(direct_pattern["scores"], key=direct_pattern["scores"].get)
        direct_pattern["confidence"] = direct_pattern["scores"][max_pattern] / total_score
        direct_pattern["pattern"] = max_pattern
    
    # If we have high confidence in the direct pattern, use it
    if direct_pattern["confidence"] > 0.7:
        pattern = direct_pattern["pattern"]
        print(f"Using enhanced pattern recognition: {pattern}")
    else:
        # Fall back to the more complex analyzer
        analyzer = ProofStructureAnalyzer()
        pattern_info = analyzer.detect_proof_pattern(theorem_text, proof_text)
        pattern = pattern_info["pattern"]
        print(f"Using enhanced pattern recognition: {pattern}")
        
    # Store linguistic features for later use
    direct_pattern["linguistic_features"] = linguistic_features
    
    # Extract additional information based on the pattern
    result_info = {
        "variables": _extract_simple_variables(theorem_text, proof_text),
        "structure_info": direct_pattern
    }
    
    # Add pattern-specific information
    if pattern == "evenness":
        # Extract the primary variable for evenness
        variables = result_info["variables"]
        result_info["variable"] = variables[0] if variables else "n"
    
    elif pattern == "induction":
        # Find the induction variable
        variables = result_info["variables"]
        induction_var = None
        
        # Look for variables in base case expressions like "n = 0"
        base_vars = re.findall(r'([a-z])\s*=\s*0', proof_text)
        if base_vars:
            induction_var = base_vars[0]
        
        result_info["variable"] = induction_var or (variables[0] if variables else "n")
    
    elif pattern == "cases":
        # Find the case variable
        variables = result_info["variables"]
        case_var = None
        
        # Look for case variable
        match = re.search(r'case\s+.*?\b([a-z])\b', proof_text, re.IGNORECASE)
        if match:
            case_var = match.group(1)
        
        result_info["case_var"] = case_var or (variables[0] if variables else "n")
    
    return pattern, result_info

def _analyze_direct_patterns(theorem_text: str, proof_text: str) -> Dict[str, Any]:
    """Analyze the proof directly for common patterns using enhanced NLP techniques."""
    # Initialize pattern scores
    pattern_scores = {
        "contradiction": 0,
        "induction": 0,
        "cases": 0,
        "direct": 0,
        "evenness": 0
    }
    
    # Process text with spaCy for deeper analysis
    theorem_doc = nlp(theorem_text)
    proof_doc = nlp(proof_text)
    
    # Extract mathematical entities
    math_entities = {
        "variables": [],
        "numbers": [],
        "operations": []
    }
    
    # Extract variables and mathematical operations
    for doc in [theorem_doc, proof_doc]:
        for token in doc:
            # Identify potential variables (single lowercase letters)
            if len(token.text) == 1 and token.text.islower() and token.text.isalpha():
                if token.text not in math_entities["variables"]:
                    math_entities["variables"].append(token.text)
            
            # Identify numbers
            if token.like_num:
                if token.text not in math_entities["numbers"]:
                    math_entities["numbers"].append(token.text)
            
            # Identify mathematical operations
            if token.text in ['+', '-', '*', '/', '=', '<', '>', '≤', '≥']:
                if token.text not in math_entities["operations"]:
                    math_entities["operations"].append(token.text)
    
    # Check for evenness pattern in theorem and proof
    if re.search(r'\b(even|divisible by 2|divisible by two)\b', theorem_text.lower()):
        pattern_scores["evenness"] += 3
    if re.search(r'\b(n \+ n|2n|2 \* n|n \* 2)\b', theorem_text.lower().replace(" ", "")):
        pattern_scores["evenness"] += 2
    if re.search(r'\b(even|divisible by 2|divisible by two)\b', proof_text.lower()):
        pattern_scores["evenness"] += 2
    if "2k" in proof_text.lower() or "2 * k" in proof_text.lower():
        pattern_scores["evenness"] += 2
        
    # Check for induction pattern
    if re.search(r'\b(induction|base case|inductive)\b', proof_text.lower()):
        pattern_scores["induction"] += 4
    if re.search(r'\b(n = 0|n=0|base case|assume for k|prove for k\+1)\b', proof_text.lower()):
        pattern_scores["induction"] += 3
    if re.search(r'\bsum\b.*\bfirst\b.*\bn\b', theorem_text.lower()):
        pattern_scores["induction"] += 2
        
    # Check for contradiction pattern
    if re.search(r'\b(contradiction|assume|contrary|absurd)\b', proof_text.lower()):
        pattern_scores["contradiction"] += 3
    if re.search(r'\b(suppose not|assume the opposite|for the sake of contradiction)\b', proof_text.lower()):
        pattern_scores["contradiction"] += 3
        
    # Check for cases pattern
    if re.search(r'\b(case|cases|first case|second case)\b', proof_text.lower()):
        pattern_scores["cases"] += 3
    if re.search(r'\b(consider|split into cases|divide into cases)\b', proof_text.lower()):
        pattern_scores["cases"] += 2
    if re.search(r'\bcase\s+\d+\b', proof_text.lower()):
        pattern_scores["cases"] += 3
    if re.search(r'\b(either|or|when|if)\b.*\b(then)\b', proof_text.lower()):
        pattern_scores["cases"] += 2
    if re.search(r'\bwe consider two cases\b', proof_text.lower()):
        pattern_scores["cases"] += 4
        
    # Direct proof is the fallback
    pattern_scores["direct"] = 1
    
    # Find the pattern with the highest score
    best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
    
    # Analyze sentence structure for deeper understanding
    proof_sentences = [sent for sent in proof_doc.sents]
    proof_structure = []
    
    for i, sent in enumerate(proof_sentences):
        # Determine the role of each sentence in the proof
        sent_type = "statement"
        
        # First sentence often contains assumptions or setup
        if i == 0 and any(token.lemma_ in ["let", "assume", "suppose"] for token in sent):
            sent_type = "assumption"
        
        # Look for conclusion indicators
        elif any(token.lemma_ in ["therefore", "thus", "hence", "conclude"] for token in sent):
            sent_type = "conclusion"
        
        # Look for case analysis indicators
        elif any(token.lemma_ in ["case", "consider"] for token in sent):
            sent_type = "case_analysis"
        
        # Look for induction indicators
        elif any(token.text.lower() in ["base", "inductive", "hypothesis"] for token in sent):
            sent_type = "induction_step"
        
        proof_structure.append({
            "text": sent.text,
            "type": sent_type,
            "entities": [token.text for token in sent if len(token.text) == 1 and token.text.islower()]
        })
    
    # Only return with reasonable confidence
    result = {
        "pattern": best_pattern[0], 
        "confidence": min(best_pattern[1]/5, 0.9),
        "scores": pattern_scores,
        "math_entities": math_entities,
        "proof_structure": proof_structure
    }
    
    if best_pattern[1] < 2:
        result["pattern"] = "direct"
        result["confidence"] = 0.5
    
    return result

def _extract_simple_variables(theorem_text: str, proof_text: str) -> List[str]:
    """Extract variables using NLP and regex techniques."""
    # Process with spaCy for linguistic analysis
    combined_doc = nlp(theorem_text + " " + proof_text)
    
    # Dictionary to track variables and their importance scores
    var_scores = {}
    
    # Find variables through linguistic analysis
    for token in combined_doc:
        # Single lowercase letters are likely variables
        if len(token.text) == 1 and token.text.islower() and token.text.isalpha():
            if token.text not in var_scores:
                var_scores[token.text] = 0
            
            # Increase score based on context
            # Variables in subject position are more important
            if token.dep_ in ["nsubj", "nsubjpass"]:
                var_scores[token.text] += 3
            # Variables that are direct objects are important
            elif token.dep_ in ["dobj", "pobj"]:
                var_scores[token.text] += 2
            # Variables that appear with numbers or mathematical operations
            elif any(child.like_num or child.text in ['+', '-', '*', '/', '='] for child in token.children):
                var_scores[token.text] += 2
            # Default score increase for any occurrence
            else:
                var_scores[token.text] += 1
    
    # Also use regex as a fallback
    regex_vars = re.findall(r'\b([a-z])\b', theorem_text + " " + proof_text)
    for var in regex_vars:
        if var not in var_scores:
            var_scores[var] = 0.5
    
    # Sort variables by importance score
    sorted_vars = sorted(var_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract just the variable names in order of importance
    return [var for var, score in sorted_vars]
