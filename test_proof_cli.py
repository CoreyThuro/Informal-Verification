#!/usr/bin/env python3
"""
CLI test script for analyzing a specific proof with the NLP-enhanced system.
"""

from patterns.enhanced_recognizer import enhanced_recognize_pattern
from patterns.nlp_analyzer import analyze_proof, get_enhanced_pattern
from translator import ProofTranslator
import json

def format_dict(d):
    """Format dictionary values to be more readable."""
    return {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in d.items()}

def test_specific_proof():
    """Test a specific proof with both analyzers."""
    
    # The test proof
    theorem = "For any positive integer n, if n² is divisible by 4, then n is divisible by 2."
    proof = """We'll prove this by contradiction. Assume that n² is divisible by 4, but n is not divisible by 2. Since n is not divisible by 2, it must be odd, so n = 2k + 1 for some integer k.

Now, let's compute n²:
n² = (2k + 1)² = 4k² + 4k + 1 = 4(k² + k) + 1

Since 4(k² + k) is clearly divisible by 4, we have n² = 4(k² + k) + 1, which means n² leaves remainder 1 when divided by 4. But this contradicts our assumption that n² is divisible by 4.

Therefore, our assumption must be false, and we conclude that if n² is divisible by 4, then n must be divisible by 2."""
    
    print("=" * 50)
    print("TESTING PROOF:")
    print(f"Theorem: {theorem}")
    print("-" * 30)
    print(f"Proof: {proof}")
    print("=" * 50)
    
    # Test with enhanced recognizer
    print("\n1. ENHANCED RECOGNIZER RESULTS:")
    pattern, pattern_info = enhanced_recognize_pattern(theorem, proof)
    
    print(f"Pattern: {pattern}")
    print(f"Confidence: {pattern_info['structure_info']['confidence']:.2f}")
    
    print("\nVariables (by importance):")
    for var in pattern_info['variables']:
        print(f"  {var}")
    
    print("\nMathematical Entities:")
    for entity_type, entities in pattern_info['structure_info']['math_entities'].items():
        if entities:
            print(f"  {entity_type}: {', '.join(entities)}")
    
    print("\nProof Structure:")
    for i, step in enumerate(pattern_info['structure_info']['proof_structure'], 1):
        print(f"  Step {i} ({step['type']}): {step['text'][:50]}...")
    
    # Test with NLP analyzer
    print("\n2. NLP ANALYZER RESULTS:")
    nlp_analysis = analyze_proof(theorem, proof)
    
    print("Pattern Scores:")
    for pattern, score in nlp_analysis['pattern_scores'].items():
        print(f"  {pattern}: {score:.2f}")
    
    print("\nVariables:")
    for var in nlp_analysis['variables']:
        print(f"  {var}")
    
    print("\nMathematical Entities:")
    for entity_type, entities in nlp_analysis['entities'].items():
        if entities:
            print(f"  {entity_type}: {', '.join(entities[:5])}")  # Show up to 5 entities
    
    print("\nProof Steps:")
    for i, step in enumerate(nlp_analysis['steps'], 1):
        print(f"  Step {i} ({step['type']}): {step['text'][:50]}...")
    
    # Test with translator
    print("\n3. TRANSLATOR RESULTS:")
    translator = ProofTranslator()
    result = translator.translate(theorem, proof)
    
    print(f"Pattern: {result['pattern']}")
    print(f"Domain: {result['domain']}")
    print(f"Verified: {result['verified']}")
    print("\nFormal Proof:")
    print(result['formal_proof'])

if __name__ == "__main__":
    test_specific_proof()
