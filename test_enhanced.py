#!/usr/bin/env python3
"""
Test script for the enhanced pattern recognition and translation system.
"""

from translator import ProofTranslator

def test_enhanced_recognition():
    """Test the enhanced pattern recognition with various examples."""
    translator = ProofTranslator()
    
    # Test 1: Simple evenness proof
    theorem1 = "For all natural numbers n, n + n is even."
    proof1 = "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
    
    result1 = translator.translate(theorem1, proof1)
    print("\n=== Test 1: Simple Evenness Proof ===")
    print(f"Pattern: {result1['pattern']}")
    print(f"Domain: {result1['domain']}")
    print(f"Verified: {result1['verified']}")
    print("\nGenerated Proof:")
    print(result1['formal_proof'])
    
    # Test 2: More complex evenness proof with divisibility language
    theorem2 = "For all integers n, the expression 2n^2 + n is divisible by 2."
    proof2 = "Let n be an integer. We have 2n^2 + n = 2n^2 + n = n(2n + 1). Since 2n is even, and n(2n + 1) = 2n·(n/2) + n = n + 2n·(n/2), we see that n(2n + 1) is the sum of n and an even number. If n is even, then n(2n + 1) is even. If n is odd, then n(2n + 1) is odd plus even, which is odd. But we know that 2n^2 is always even, so 2n^2 + n is even if and only if n is even. Therefore, 2n^2 + n is divisible by 2."
    
    result2 = translator.translate(theorem2, proof2)
    print("\n=== Test 2: Complex Evenness Proof with Divisibility ===")
    print(f"Pattern: {result2['pattern']}")
    print(f"Domain: {result2['domain']}")
    print(f"Verified: {result2['verified']}")
    print("\nGenerated Proof:")
    print(result2['formal_proof'])
    
    # Test 3: Induction proof
    theorem3 = "For all natural numbers n, the sum of the first n natural numbers is n(n+1)/2."
    proof3 = "We proceed by induction on n. For the base case, when n = 0, the sum is 0, and 0(0+1)/2 = 0, so the formula holds. Assume that for some k ≥ 0, the sum of the first k natural numbers is k(k+1)/2. We need to prove that the sum of the first k+1 natural numbers is (k+1)(k+2)/2. The sum of the first k+1 natural numbers is (the sum of the first k natural numbers) + (k+1). By the induction hypothesis, this is k(k+1)/2 + (k+1) = k(k+1)/2 + 2(k+1)/2 = (k+1)(k+2)/2. Therefore, the formula holds for all natural numbers n."
    
    result3 = translator.translate(theorem3, proof3)
    print("\n=== Test 3: Induction Proof ===")
    print(f"Pattern: {result3['pattern']}")
    print(f"Domain: {result3['domain']}")
    print(f"Verified: {result3['verified']}")
    print("\nGenerated Proof:")
    print(result3['formal_proof'])

if __name__ == "__main__":
    test_enhanced_recognition()
