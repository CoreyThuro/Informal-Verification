"""
Tests for the main translator functionality.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translator import ProofTranslator

class TestTranslator(unittest.TestCase):
    """Tests for the ProofTranslator class."""
    
    def setUp(self):
        """Set up the translator."""
        self.translator = ProofTranslator()
    
    def test_evenness_translation(self):
        """Test translation of an evenness proof."""
        theorem = "For all natural numbers n, n + n is even."
        proof = "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
        
        result = self.translator.translate(theorem, proof)
        
        self.assertEqual(result["pattern"], "evenness")
        self.assertIn("exists", result["formal_proof"])
        self.assertIn("ring", result["formal_proof"])
    
    def test_induction_translation(self):
        """Test translation of an induction proof."""
        theorem = "For all natural numbers n, the sum 0 + 1 + ... + n equals n(n+1)/2."
        proof = """
        We prove this by induction on n.
        Base case: When n = 0, the sum is 0, and 0(0+1)/2 = 0, so the formula holds.
        Inductive step: Assume the formula holds for k, so 0 + 1 + ... + k = k(k+1)/2.
        Then for n = k+1, we have 0 + 1 + ... + k + (k+1) = k(k+1)/2 + (k+1).
        This simplifies to (k+1)(k/2 + 1) = (k+1)(k+2)/2, which is the formula for n = k+1.
        Thus, by induction, the formula holds for all natural numbers n.
        """
        
        result = self.translator.translate(theorem, proof)
        
        self.assertEqual(result["pattern"], "induction")
        self.assertIn("induction", result["formal_proof"])
    
    def test_contradiction_translation(self):
        """Test translation of a contradiction proof."""
        theorem = "There is no rational number r such that r² = 2."
        proof = """
        Suppose by contradiction that there is a rational number r such that r² = 2.
        Then r can be written as a/b where a and b are integers with no common factors.
        So (a/b)² = 2, which means a² = 2b². This implies that a² is even, which means a is even.
        So a = 2c for some integer c. Substituting, we get (2c)² = 2b², so 4c² = 2b², thus b² = 2c².
        This means b² is even, so b is even. But this contradicts our assumption that a and b have no common factors.
        Therefore, there is no rational number r such that r² = 2.
        """
        
        result = self.translator.translate(theorem, proof)
        
        self.assertEqual(result["pattern"], "contradiction")
        self.assertIn("contradiction", result["formal_proof"])
    
    def test_domain_detection(self):
        """Test domain detection for different proofs."""
        # Number theory
        theorem1 = "For all integers n, if n is even, then n² is even."
        proof1 = "Let n be an even integer. Then n = 2k for some integer k. Thus, n² = (2k)² = 4k², which is even."
        result1 = self.translator.translate(theorem1, proof1)
        self.assertEqual(result1["domain"], "11")  # Number theory
        
        # Algebra
        theorem2 = "For all matrices A and B, if A and B are invertible, then AB is invertible."
        proof2 = "Let A and B be invertible matrices. Consider the product AB and the product B⁻¹A⁻¹. We have AB·B⁻¹A⁻¹ = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I. Thus AB is invertible with inverse B⁻¹A⁻¹."
        result2 = self.translator.translate(theorem2, proof2)
        self.assertEqual(result2["domain"], "12-20")  # Algebra
    
    def test_verification(self):
        """Test that verification works for valid proofs."""
        theorem = "For all natural numbers n, n + n is even."
        proof = "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
        
        result = self.translator.translate(theorem, proof)
        
        # Note: This test will only pass if Coq is installed and available
        # If not, it will be skipped with a warning
        if result["verified"] is False and "Coq compiler not found" in result.get("error_message", ""):
            self.skipTest("Coq compiler not found. Skipping verification test.")
        
        self.assertTrue(result["verified"])

if __name__ == "__main__":
    unittest.main()