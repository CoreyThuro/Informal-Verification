"""
Test script for the knowledge graph integration.

This script tests the functionality of the mathematical knowledge graph
and its integration with the NLP analyzer.
"""

import sys
import json
from knowledge.knowledge_graph import default_graph
from knowledge.graph_analyzer import GraphEnhancedAnalyzer

def test_knowledge_graph_analysis():
    """Test the knowledge graph enhanced analysis on example theorems and proofs."""
    # Initialize graph-enhanced analyzer
    graph_analyzer = GraphEnhancedAnalyzer(default_graph)
    
    # Example theorems and proofs
    examples = [
        {
            "name": "Evenness proof",
            "theorem": "If n is an even number, then n^2 is also even.",
            "proof": "Since n is even, we can write n = 2k for some integer k. Then n^2 = (2k)^2 = 4k^2 = 2(2k^2). Since 2k^2 is an integer, n^2 = 2(2k^2) is even by definition."
        },
        {
            "name": "Sum of odd numbers",
            "theorem": "The sum of two odd numbers is even.",
            "proof": "Let a and b be odd numbers. Then a = 2m + 1 and b = 2n + 1 for some integers m and n. The sum a + b = 2m + 1 + 2n + 1 = 2(m + n + 1). Since m + n + 1 is an integer, a + b is even."
        },
        {
            "name": "Division theorem",
            "theorem": "For any integers a and b with b > 0, there exist unique integers q and r such that a = bq + r and 0 ≤ r < b.",
            "proof": "We can use induction on a. For the base case, if a < b, then q = 0 and r = a satisfies the conditions. For the inductive step, assume the theorem holds for some a ≥ b. Then for a + 1, we have a + 1 = bq + r + 1. If r + 1 < b, then a + 1 = bq + (r + 1) and we're done. If r + 1 = b, then a + 1 = bq + b = b(q + 1) + 0, which also satisfies the conditions."
        }
    ]
    
    # Test each example
    for example in examples:
        print(f"\n\n=== Testing: {example['name']} ===")
        print(f"Theorem: {example['theorem']}")
        print(f"Proof: {example['proof']}")
        
        # Perform knowledge graph analysis
        analysis = graph_analyzer.analyze_proof(example['theorem'], example['proof'])
        
        # Print key results
        print("\n--- Knowledge Graph Analysis Results ---")
        print(f"Pattern: {analysis.get('pattern', 'Unknown')} (Confidence: {analysis.get('confidence', 0.0):.2f})")
        
        # Print identified concepts
        print("\nTheorem Concepts:")
        for concept in analysis.get('theorem_concepts', []):
            print(f"  - {concept['text']} ({concept['concept']})")
        
        print("\nProof Concepts:")
        for concept in analysis.get('proof_concepts', [])[:5]:  # Limit to first 5 for brevity
            print(f"  - {concept['text']} ({concept['concept']})")
        
        # Print related theorems
        related_theorems = graph_analyzer.suggest_related_theorems(example['theorem'], example['proof'])
        print("\nRelated Theorems:")
        for theorem in related_theorems:
            print(f"  - {theorem['theorem']} (Relevance: {theorem['relevance']:.2f})")
            print(f"    Statement: {theorem['statement']}")
            print(f"    Common concepts: {', '.join(theorem['common_concepts'])}")
        
        # Get concept explanations
        print("\nConcept Explanations:")
        concepts = graph_analyzer.get_concept_explanations(example['theorem'])
        for concept in concepts:
            if 'explanation' in concept and concept['explanation']:
                print(f"  - {concept['text']} ({concept['concept']}): {concept['explanation']}")

if __name__ == "__main__":
    test_knowledge_graph_analysis()
