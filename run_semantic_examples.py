#!/usr/bin/env python3
"""
Script to run the semantic examples from the examples/semantic_examples.txt file.
"""

import re
from translator import ProofTranslator

def parse_examples_file(file_path):
    """Parse the examples file to extract theorems and proofs."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split the content into examples
    examples = re.split(r'## Example \d+:', content)[1:]
    
    results = []
    for example in examples:
        # Extract the title
        title_match = re.search(r'(.*?)\n', example.strip())
        title = title_match.group(1).strip() if title_match else "Unknown Example"
        
        # Extract theorem and proof
        theorem_match = re.search(r'### Theorem\n(.*?)\n\n### Proof', example, re.DOTALL)
        proof_match = re.search(r'### Proof\n(.*?)(?:\n\n##|\Z)', example, re.DOTALL)
        
        if theorem_match and proof_match:
            theorem = theorem_match.group(1).strip()
            proof = proof_match.group(1).strip()
            results.append((title, theorem, proof))
    
    return results

def run_examples():
    """Run the semantic examples through the translator."""
    examples = parse_examples_file('examples/semantic_examples.txt')
    translator = ProofTranslator()
    
    for i, (title, theorem, proof) in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}: {title}")
        print(f"{'='*80}")
        
        print("Theorem:")
        print(theorem)
        print("\nProof:")
        print(proof)
        
        # Translate the proof
        result = translator.translate(theorem, proof)
        
        print("\nResults:")
        print(f"Pattern: {result['pattern']}")
        print(f"Domain: {result['domain']}")
        print(f"Verified: {result['verified']}")
        print("\nGenerated Coq Proof:")
        print(result['formal_proof'])
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    run_examples()
