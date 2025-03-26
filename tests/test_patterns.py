"""
Tests for pattern recognition.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patterns.recognizer import recognize_pattern

class TestPatternRecognition(unittest.TestCase):
    """Tests for pattern recognition module."""
    
    def test_evenness_recognition(self):
        """Test recognition of evenness proofs."""
        # Clear evenness proof
        theorem1 = "For all natural numbers n, n + n is even."
        proof1 = "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
        pattern1, info1 = recognize_pattern(theorem1, proof1)
        self.assertEqual(pattern1, "evenness")
        self.assertEqual(info1["variable"], "n")
        
        # Slightly less explicit
        theorem2 = "For any integer x, the sum x + x is divisible by 2."
        proof2 = "For any integer x, we have x + x = 2x, which is divisible by 2."
        pattern2, info2 = recognize_pattern(theorem2, proof2)
        self.assertEqual(pattern2, "evenness")
        self.assertEqual(info2["variable"], "x")
    
    def test_induction_recognition(self):
        """Test recognition of induction proofs."""
        theorem = "For all natural numbers n, the sum 1 + 2 + ... + n equals n(n+1)/2."
        proof = """
        We prove this by induction on n.
        Base case: When n = 1, the sum is 1, and 1(1+1)/2 = 1, so the formula holds.
        Inductive step: Assume the formula holds for some k, so 1 + 2 + ... + k = k(k+1)/2.
        We need to show it holds for n = k+1.
        We have 1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2,
        which is the formula for n = k+1.
        Thus, by induction, the formula holds for all natural numbers n.
        """
        pattern, info = recognize_pattern(theorem, proof)
        self.assertEqual(pattern, "induction")
        self.assertEqual(info["variable"], "n")
    
    def test_contradiction_recognition(self):
        """Test recognition of contradiction proofs."""
        theorem = "The square root of 2 is irrational."
        proof = """
        Suppose, by contradiction, that √2 is rational.
        Then √2 = a/b for some integers a and b with gcd(a,b) = 1.
        Squaring both sides, we get 2 = a²/b².
        Thus, a² = 2b².
        This means a² is even, so a is even. Let a = 2c.
        Then 2b² = (2c)² = 4c², so b² = 2c².
        This means b² is even, so b is even.
        But this contradicts our assumption that gcd(a,b) = 1.
        Therefore, √2 is irrational.
        """
        pattern, info = recognize_pattern(theorem, proof)
        self.assertEqual(pattern, "contradiction")
    
    def test_cases_recognition(self):
        """Test recognition of case analysis proofs."""
        theorem = "For any integer n, n² - n is even."
        proof = """
        We consider two cases:
        Case 1: n is even. Then n = 2k for some integer k.
        So n² - n = (2k)² - 2k = 4k² - 2k = 2(2k² - k), which is even.
        Case 2: n is odd. Then n = 2k + 1 for some integer k.
        So n² - n = (2k + 1)² - (2k + 1) = 4k² + 4k + 1 - 2k - 1 = 4k² + 2k = 2(2k² + k), which is even.
        In both cases, n² - n is even, completing the proof.
        """
        pattern, info = recognize_pattern(theorem, proof)
        self.assertEqual(pattern, "cases")
    
    def test_direct_proof_recognition(self):
        """Test recognition of direct proofs."""
        theorem = "For all real numbers a and b, (a+b)² = a² + 2ab + b²."
        proof = """
        Let a and b be real numbers.
        Then (a+b)² = (a+b)(a+b) = a(a+b) + b(a+b) = a² + ab + ba + b² = a² + 2ab + b².
        """
        pattern, info = recognize_pattern(theorem, proof)
        self.assertEqual(pattern, "direct")
        self.assertIn("a", info["variables"])
        self.assertIn("b", info["variables"])

if __name__ == "__main__":
    unittest.main()