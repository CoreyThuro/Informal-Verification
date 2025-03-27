"""
NLP-based analyzer for mathematical proofs using spaCy.

This module enhances pattern recognition by applying natural language processing
techniques to extract semantic information from mathematical proofs.
"""

import re
import spacy
from typing import Dict, List, Tuple, Any, Set, Optional

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model isn't installed, download it
    import subprocess
    import sys
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Mathematical entity patterns
MATH_ENTITIES = {
    "VARIABLE": r"[a-zA-Z](?:\([a-zA-Z0-9,\s]+\))?",
    "NUMBER": r"\d+",
    "OPERATION": r"[+\-*/=<>≤≥≠]",
    "FUNCTION": r"[a-zA-Z]+\([a-zA-Z0-9,\s]+\)",
    "QUANTIFIER": r"\b(for all|exists|forall|∀|∃)\b",
    "LOGICAL_OP": r"\b(and|or|not|implies|if|then|iff|⇒|⇔|∧|∨|¬)\b",
}

# Mathematical keywords for different proof types
PROOF_KEYWORDS = {
    "induction": [
        "induction", "base case", "inductive step", "inductive hypothesis",
        "assume", "show", "holds for", "k+1", "n+1"
    ],
    "contradiction": [
        "contradiction", "assume", "contrary", "false", "impossible",
        "absurd", "leads to", "cannot be"
    ],
    "cases": [
        "case", "cases", "consider", "either", "or", "when", "if", "then"
    ],
    "evenness": [
        "even", "odd", "divisible by 2", "divisible by two", "multiple of 2",
        "multiple of two", "2k", "2k+1"
    ],
    "direct": [
        "directly", "straightforward", "immediately", "clearly", "observe"
    ]
}

def analyze_proof(theorem_text: str, proof_text: str) -> Dict[str, Any]:
    """
    Analyze a mathematical proof using NLP techniques.
    
    Args:
        theorem_text: The theorem statement
        proof_text: The proof text
        
    Returns:
        A dictionary containing extracted information:
        - pattern_scores: Scores for different proof patterns
        - entities: Mathematical entities found
        - dependencies: Key dependency relationships
        - steps: Identified proof steps
        - variables: Extracted variables
    """
    # Process the texts with spaCy
    theorem_doc = nlp(theorem_text)
    proof_doc = nlp(proof_text)
    
    # Extract mathematical entities
    entities = extract_math_entities(theorem_text, proof_text)
    
    # Identify proof pattern scores using NLP features
    pattern_scores = score_patterns_nlp(theorem_doc, proof_doc)
    
    # Extract variables
    variables = extract_variables(theorem_doc, proof_doc, entities)
    
    # Identify proof steps
    steps = identify_proof_steps(proof_doc)
    
    # Analyze dependencies
    dependencies = analyze_dependencies(theorem_doc, proof_doc)
    
    return {
        "pattern_scores": pattern_scores,
        "entities": entities,
        "dependencies": dependencies,
        "steps": steps,
        "variables": variables
    }

def extract_math_entities(theorem_text: str, proof_text: str) -> Dict[str, List[str]]:
    """Extract mathematical entities from the theorem and proof texts."""
    combined_text = theorem_text + " " + proof_text
    entities = {entity_type: [] for entity_type in MATH_ENTITIES}
    
    # Extract entities using regex patterns
    for entity_type, pattern in MATH_ENTITIES.items():
        matches = re.finditer(pattern, combined_text)
        for match in matches:
            entity = match.group(0)
            if entity not in entities[entity_type]:
                entities[entity_type].append(entity)
    
    return entities

def score_patterns_nlp(theorem_doc, proof_doc) -> Dict[str, float]:
    """
    Score different proof patterns using NLP features.
    
    This function uses linguistic features like part-of-speech tags,
    dependencies, and entity recognition to score different proof patterns.
    """
    pattern_scores = {
        "induction": 0.0,
        "contradiction": 0.0,
        "cases": 0.0,
        "evenness": 0.0,
        "direct": 0.0
    }
    
    # Combine theorem and proof text for analysis
    combined_text = theorem_doc.text + " " + proof_doc.text
    combined_text_lower = combined_text.lower()
    
    # Score based on keywords
    for pattern, keywords in PROOF_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in combined_text_lower:
                pattern_scores[pattern] += 1.0
    
    # Score based on linguistic features
    for token in proof_doc:
        # Check for induction indicators
        if token.lemma_ in ["assume", "induct", "base", "step", "hypothesis"]:
            pattern_scores["induction"] += 0.5
        
        # Check for contradiction indicators
        if token.lemma_ in ["contradict", "false", "impossible", "absurd"]:
            pattern_scores["contradiction"] += 0.5
        
        # Check for case analysis indicators
        if token.lemma_ in ["case", "consider", "either", "when"]:
            pattern_scores["cases"] += 0.5
        
        # Check for evenness indicators
        if token.lemma_ in ["even", "odd", "divisible", "multiple"]:
            pattern_scores["evenness"] += 0.5
    
    # Check for specific sentence structures
    for sent in proof_doc.sents:
        # Induction structure
        if "base case" in sent.text.lower() or "inductive step" in sent.text.lower():
            pattern_scores["induction"] += 2.0
        
        # Contradiction structure
        if "assume" in sent.text.lower() and any(word in sent.text.lower() for word in ["contradiction", "contrary"]):
            pattern_scores["contradiction"] += 2.0
        
        # Cases structure
        if "case" in sent.text.lower() and any(token.text.isdigit() for token in sent):
            pattern_scores["cases"] += 2.0
        
        # Evenness structure
        if "divisible by 2" in sent.text.lower() or "multiple of 2" in sent.text.lower():
            pattern_scores["evenness"] += 2.0
    
    return pattern_scores

def extract_variables(theorem_doc, proof_doc, entities: Dict[str, List[str]]) -> List[str]:
    """Extract variables from the theorem and proof using NLP."""
    variables = []
    
    # Extract from entities
    if "VARIABLE" in entities:
        variables.extend(entities["VARIABLE"])
    
    # Extract using dependency parsing
    for doc in [theorem_doc, proof_doc]:
        for token in doc:
            # Look for tokens that might be variables
            if token.is_alpha and len(token.text) == 1 and token.text.islower():
                # Check if it's not part of a larger word
                if token.is_sent_start or not token.nbor(-1).is_alpha:
                    if token.text not in variables:
                        variables.append(token.text)
    
    return variables

def identify_proof_steps(proof_doc) -> List[Dict[str, Any]]:
    """
    Identify and categorize steps in the proof.
    
    Returns a list of steps, each with:
    - text: The step text
    - type: The step type (assumption, deduction, etc.)
    - entities: Entities involved in this step
    """
    steps = []
    
    # Split into sentences as potential steps
    for i, sent in enumerate(proof_doc.sents):
        step_type = "statement"  # Default type
        
        # Try to determine step type
        if i == 0 and any(token.lemma_ in ["let", "assume", "suppose"] for token in sent):
            step_type = "assumption"
        elif any(token.lemma_ in ["therefore", "thus", "hence"] for token in sent):
            step_type = "conclusion"
        elif any(token.lemma_ in ["since", "because", "as"] for token in sent):
            step_type = "justification"
        elif any(token.lemma_ in ["if", "when", "case"] for token in sent):
            step_type = "case_analysis"
        
        # Extract entities in this step
        step_entities = []
        for token in sent:
            if token.is_alpha and len(token.text) == 1 and token.text.islower():
                step_entities.append(token.text)
        
        steps.append({
            "text": sent.text,
            "type": step_type,
            "entities": step_entities
        })
    
    return steps

def analyze_dependencies(theorem_doc, proof_doc) -> List[Dict[str, str]]:
    """
    Analyze dependency relationships in the theorem and proof.
    
    This helps understand the relationships between mathematical objects.
    """
    dependencies = []
    
    for doc in [theorem_doc, proof_doc]:
        for token in doc:
            # Focus on relationships that might be mathematically relevant
            if token.dep_ in ["nsubj", "dobj", "pobj", "attr"]:
                if token.head.pos_ in ["VERB", "AUX"]:
                    dependencies.append({
                        "source": token.text,
                        "relation": token.head.text,
                        "target": token.head.text,
                        "dep_type": token.dep_
                    })
    
    return dependencies

def get_enhanced_pattern(theorem_text: str, proof_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get the enhanced pattern recognition result using NLP techniques.
    
    Args:
        theorem_text: The theorem statement
        proof_text: The proof text
        
    Returns:
        A tuple of (pattern_name, pattern_info)
    """
    # Analyze the proof using NLP
    analysis = analyze_proof(theorem_text, proof_text)
    
    # Get pattern scores
    pattern_scores = analysis["pattern_scores"]
    
    # Find the highest scoring pattern
    max_score = 0
    best_pattern = "direct"  # Default to direct proof
    
    for pattern, score in pattern_scores.items():
        if score > max_score:
            max_score = score
            best_pattern = pattern
    
    # Prepare pattern info
    pattern_info = {
        "variables": analysis["variables"],
        "steps": analysis["steps"],
        "entities": analysis["entities"],
        "nlp_confidence": max_score / sum(pattern_scores.values()) if sum(pattern_scores.values()) > 0 else 0
    }
    
    return best_pattern, pattern_info
