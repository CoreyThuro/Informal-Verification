#!/usr/bin/env python3
"""
Test script for the NLP-enhanced pattern recognition system.
This script tests the advanced NLP capabilities added to the pattern recognition system.
"""

import json
from patterns.enhanced_recognizer import enhanced_recognize_pattern
from patterns.nlp_analyzer import analyze_proof, get_enhanced_pattern
from translator import ProofTranslator

def test_nlp_enhanced_recognition():
    """Test the NLP-enhanced pattern recognition with various examples."""
    translator = ProofTranslator()
    
    # Load examples from semantic_examples.txt
    examples = load_examples_from_file("examples/semantic_examples.txt")
    
    for i, example in enumerate(examples, 1):
        print(f"\n=== Example {i}: {example['title']} ===")
        
        # Test the enhanced recognizer
        pattern, pattern_info = enhanced_recognize_pattern(example['theorem'], example['proof'])
        
        print(f"Enhanced Recognizer Pattern: {pattern}")
        print(f"Confidence: {pattern_info['structure_info']['confidence']:.2f}")
        
        # Print the top variables with their importance
        print("\nVariables (by importance):")
        for var in pattern_info['variables'][:3]:  # Show top 3 variables
            print(f"  {var}")
        
        # Test the NLP analyzer
        nlp_analysis = analyze_proof(example['theorem'], example['proof'])
        
        print("\nNLP Analysis:")
        print(f"  Pattern Scores: {format_dict(nlp_analysis['pattern_scores'])}")
        
        # Print mathematical entities
        print("\nMathematical Entities:")
        for entity_type, entities in nlp_analysis['entities'].items():
            if entities:
                print(f"  {entity_type}: {', '.join(entities[:3])}")  # Show up to 3 entities
        
        # Print proof steps
        print("\nProof Steps:")
        for j, step in enumerate(nlp_analysis['steps'][:2], 1):  # Show first 2 steps
            print(f"  Step {j} ({step['type']}): {step['text'][:50]}...")
        
        # Also test the translator
        result = translator.translate(example['theorem'], example['proof'])
        print(f"\nTranslator Pattern: {result['pattern']}")
        print(f"Domain: {result['domain']}")
        print(f"Verified: {result['verified']}")
        
        print("\n" + "-"*50)

def load_examples_from_file(filename):
    """Load examples from the semantic examples file."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by example markers
    example_sections = content.split("## Example")[1:]
    
    examples = []
    for section in example_sections:
        lines = section.strip().split('\n')
        
        # Extract title
        title = lines[0].strip(": ")
        
        # Find theorem and proof sections
        theorem_start = section.find("### Theorem")
        proof_start = section.find("### Proof")
        next_section = section.find("##", proof_start)
        if next_section == -1:
            next_section = len(section)
        
        # Extract theorem and proof text
        theorem = section[theorem_start + len("### Theorem"):proof_start].strip()
        proof = section[proof_start + len("### Proof"):next_section].strip()
        
        examples.append({
            "title": title,
            "theorem": theorem,
            "proof": proof
        })
    
    return examples

def format_dict(d):
    """Format dictionary values to be more readable."""
    return {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in d.items()}

def test_specific_examples():
    """Test specific examples to showcase NLP capabilities."""
    
    # Example with complex linguistic structure
    theorem = "If a function f is continuous on a closed interval [a,b] and differentiable on (a,b), then there exists c in (a,b) such that f(b) - f(a) = f'(c)(b-a)."
    proof = "Let F(x) = f(x) - f(a) - (f(b) - f(a))/(b-a) * (x-a). Then F is continuous on [a,b] and differentiable on (a,b). Also, F(a) = F(b) = 0. By Rolle's theorem, there exists c in (a,b) such that F'(c) = 0. Computing F'(c), we get f'(c) - (f(b) - f(a))/(b-a) = 0, which gives us f(b) - f(a) = f'(c)(b-a)."
    
    print("\n=== Testing Mean Value Theorem Example ===")
    
    # Test enhanced recognizer
    pattern, pattern_info = enhanced_recognize_pattern(theorem, proof)
    print(f"Enhanced Recognizer Pattern: {pattern}")
    print(f"Confidence: {pattern_info['structure_info']['confidence']:.2f}")
    
    # Test NLP analyzer
    nlp_analysis = analyze_proof(theorem, proof)
    
    # Print proof structure analysis
    print("\nProof Structure Analysis:")
    for i, step in enumerate(pattern_info['structure_info']['proof_structure'], 1):
        print(f"  Step {i} ({step['type']}): {step['text'][:50]}...")
    
    # Print mathematical entities
    print("\nMathematical Entities:")
    for entity_type, entities in pattern_info['structure_info']['math_entities'].items():
        if entities:
            print(f"  {entity_type}: {', '.join(entities)}")

if __name__ == "__main__":
    print("Testing NLP-Enhanced Pattern Recognition")
    print("=======================================")
    test_nlp_enhanced_recognition()
    test_specific_examples()
