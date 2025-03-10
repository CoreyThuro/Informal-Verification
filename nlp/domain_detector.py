"""
Domain detector for mathematical proofs.
Identifies the mathematical field and relevant concepts in a proof.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise ImportError("spaCy model 'en_core_web_sm' not found. Please install with 'python -m spacy download en_core_web_sm'")

# MSC (Mathematics Subject Classification) categories
MSC_CATEGORIES = {
    "00": "General Mathematics",
    "01": "History and Biography",
    "03": "Mathematical Logic and Foundations",
    "05": "Combinatorics",
    "06": "Order, Lattices, Ordered Algebraic Structures",
    "08": "General Algebraic Systems",
    "11": "Number Theory",
    "12": "Field Theory and Polynomials",
    "13": "Commutative Algebra",
    "14": "Algebraic Geometry",
    "15": "Linear and Multilinear Algebra; Matrix Theory",
    "16": "Associative Rings and Algebras",
    "17": "Nonassociative Rings and Algebras",
    "18": "Category Theory; Homological Algebra",
    "19": "K-Theory",
    "20": "Group Theory and Generalizations",
    "22": "Topological Groups, Lie Groups",
    "26": "Real Functions",
    "28": "Measure and Integration",
    "30": "Functions of a Complex Variable",
    "31": "Potential Theory",
    "32": "Several Complex Variables and Analytic Spaces",
    "33": "Special Functions",
    "34": "Ordinary Differential Equations",
    "35": "Partial Differential Equations",
    "37": "Dynamical Systems and Ergodic Theory",
    "39": "Difference and Functional Equations",
    "40": "Sequences, Series, Summability",
    "41": "Approximations and Expansions",
    "42": "Fourier Analysis",
    "43": "Abstract Harmonic Analysis",
    "44": "Integral Transforms, Operational Calculus",
    "45": "Integral Equations",
    "46": "Functional Analysis",
    "47": "Operator Theory",
    "49": "Calculus of Variations and Optimal Control",
    "51": "Geometry",
    "52": "Convex and Discrete Geometry",
    "53": "Differential Geometry",
    "54": "General Topology",
    "55": "Algebraic Topology",
    "57": "Manifolds and Cell Complexes",
    "58": "Global Analysis, Analysis on Manifolds",
    "60": "Probability Theory and Stochastic Processes",
    "62": "Statistics",
    "65": "Numerical Analysis",
    "68": "Computer Science",
    "70": "Mechanics of Particles and Systems",
    "74": "Mechanics of Deformable Solids",
    "76": "Fluid Mechanics",
    "78": "Optics, Electromagnetic Theory",
    "80": "Classical Thermodynamics, Heat Transfer",
    "81": "Quantum Theory",
    "82": "Statistical Mechanics, Structure of Matter",
    "83": "Relativity and Gravitational Theory",
    "85": "Astronomy and Astrophysics",
    "86": "Geophysics",
    "90": "Operations Research, Mathematical Programming",
    "91": "Game Theory, Economics, Finance, and Social Sciences",
    "92": "Biology and Other Natural Sciences",
    "93": "Systems Theory; Control",
    "94": "Information and Communication Theory, Circuits",
    "97": "Mathematics Education"
}

# Domain-specific keyword mappings
DOMAIN_KEYWORDS = {
    # Number Theory
    "11": ["prime", "number", "integer", "divisor", "divisible", "modulo", "congruence", 
           "gcd", "lcm", "factor", "remainder", "quotient", "coprime", "prime", 
           "composite", "even", "odd", "natural"],
    
    # Algebra
    "12-20": ["group", "field", "ring", "algebra", "module", "polynomial", "matrix", 
            "determinant", "eigenvalue", "vector", "basis", "linear", "algebra", 
            "homomorphism", "isomorphism", "subgroup", "permutation", "symmetry"],
    
    # Analysis
    "26-42": ["limit", "continuity", "differentiable", "integral", "derivative", "series", 
             "sequence", "convergence", "function", "real", "complex", "measure", 
             "bounded", "continuous", "uniformly", "supremum", "infimum"],
    
    # Topology
    "54-55": ["open", "closed", "compact", "connected", "hausdorff", "neighborhood", 
              "continuous", "homeomorphism", "topology", "metric", "space", "cover", 
              "manifold", "boundary", "interior", "closure"],
    
    # Geometry
    "51-53": ["line", "plane", "angle", "triangle", "circle", "polygon", "distance", 
              "coordinate", "geometric", "euclidean", "sphere", "curve", "surface", 
              "polygon", "polyhedron", "tangent"],
    
    # Set Theory and Logic
    "03": ["set", "subset", "element", "belongs", "union", "intersection", "complement", 
           "cartesian", "product", "relation", "function", "logical", "proposition", 
           "true", "false", "implies", "equivalent"]
}

class DomainDetector:
    """
    Detector for mathematical domains in proofs.
    """
    
    def __init__(self):
        """Initialize the domain detector."""
        # Flatten the domain keywords for easier lookup
        self.domain_to_keywords = DOMAIN_KEYWORDS
        self.keyword_to_domain = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in self.keyword_to_domain:
                    self.keyword_to_domain[keyword].append(domain)
                else:
                    self.keyword_to_domain[keyword] = [domain]
    
    def detect_domain(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Detect the mathematical domain of a theorem and its proof.
        
        Args:
            theorem_text: The theorem statement
            proof_text: The proof text
            
        Returns:
            A dictionary with domain information
        """
        # Combine text for analysis
        combined_text = f"{theorem_text} {proof_text}".lower()
        
        # Count domain keywords
        domain_scores = self._count_domain_keywords(combined_text)
        
        # Determine primary domain
        primary_domain, domain_confidence = self._determine_primary_domain(domain_scores)
        
        # Get MSC code
        msc_code = self._get_msc_code(primary_domain)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(combined_text, primary_domain)
        
        # Check for specific proof techniques
        proof_techniques = self._identify_proof_techniques(proof_text)
        
        return {
            "primary_domain": primary_domain,
            "confidence": domain_confidence,
            "msc_code": msc_code,
            "msc_name": MSC_CATEGORIES.get(msc_code.split('-')[0], "General Mathematics"),
            "domain_scores": domain_scores,
            "key_concepts": key_concepts,
            "proof_techniques": proof_techniques
        }
    
    def _count_domain_keywords(self, text: str) -> Dict[str, int]:
        """
        Count occurrences of domain-specific keywords in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary mapping domains to their scores
        """
        # Initialize scores for each domain
        domain_scores = {domain: 0 for domain in self.domain_to_keywords}
        
        # Count keyword occurrences
        for domain, keywords in self.domain_to_keywords.items():
            for keyword in keywords:
                # Count the keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
                domain_scores[domain] += count
        
        return domain_scores
    
    def _determine_primary_domain(self, domain_scores: Dict[str, int]) -> Tuple[str, float]:
        """
        Determine the primary domain based on keyword scores.
        
        Args:
            domain_scores: Dictionary mapping domains to their scores
            
        Returns:
            Tuple of (primary_domain, confidence)
        """
        # Find the domain with the highest score
        max_score = max(domain_scores.values())
        
        if max_score == 0:
            # No domain keywords found
            return "general", 0.0
        
        # Get domains with the highest score
        top_domains = [domain for domain, score in domain_scores.items() if score == max_score]
        
        if len(top_domains) == 1:
            # Clear winner
            primary_domain = top_domains[0]
        else:
            # Multiple domains tied for first place
            # Use the first one (arbitrary choice, could be improved)
            primary_domain = top_domains[0]
        
        # Calculate confidence
        total_score = sum(domain_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        return primary_domain, confidence
    
    def _get_msc_code(self, domain: str) -> str:
        """
        Get the MSC code for a domain.
        
        Args:
            domain: The domain
            
        Returns:
            The MSC code
        """
        return domain
    
    def _extract_key_concepts(self, text: str, primary_domain: str) -> List[str]:
        """
        Extract key mathematical concepts from the text.
        
        Args:
            text: The text to analyze
            primary_domain: The primary domain
            
        Returns:
            A list of key concepts
        """
        # Get domain keywords
        if primary_domain in self.domain_to_keywords:
            domain_keywords = self.domain_to_keywords[primary_domain]
        else:
            # Use all keywords if domain not recognized
            domain_keywords = [kw for kws in self.domain_to_keywords.values() for kw in kws]
        
        # Find occurrences of domain-specific concepts in the text
        concepts = []
        for keyword in domain_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                concepts.append(keyword)
        
        # Extract expressions like "x + y", "f(x)", etc.
        # Simple regex to find mathematical expression patterns
        expr_patterns = [
            r'\b([a-zA-Z])(?:\s*[\+\-\*\/\^]\s*([a-zA-Z0-9]+))+\b',  # x + y, a * b
            r'\b([a-zA-Z])\s*\(\s*([a-zA-Z](?:,\s*[a-zA-Z])*)\s*\)\b',  # f(x), g(x,y)
            r'\b([a-zA-Z]+)\s*=\s*([a-zA-Z0-9]+(?:\s*[\+\-\*\/\^]\s*[a-zA-Z0-9]+)*)\b'  # x = y + z
        ]
        
        for pattern in expr_patterns:
            expressions = re.findall(pattern, text)
            if expressions:
                for expr in expressions:
                    if isinstance(expr, tuple):
                        concepts.append(''.join(expr))
                    else:
                        concepts.append(expr)
        
        # Remove duplicates and sort
        return sorted(list(set(concepts)))
    
    def _identify_proof_techniques(self, proof_text: str) -> List[str]:
        """
        Identify proof techniques used in the proof.
        
        Args:
            proof_text: The proof text
            
        Returns:
            A list of identified proof techniques
        """
        techniques = []
        
        # Look for proof by induction
        if re.search(r'\b(induction|base case|inductive|hypothesis)\b', proof_text, re.IGNORECASE):
            techniques.append("induction")
        
        # Look for proof by contradiction
        if re.search(r'\b(contradiction|contrary|suppose not|assume not)\b', proof_text, re.IGNORECASE):
            techniques.append("contradiction")
        
        # Look for proof by contrapositive
        if re.search(r'\b(contrapositive|not.*then not)\b', proof_text, re.IGNORECASE):
            techniques.append("contrapositive")
        
        # Look for proof by cases
        if re.search(r'\b(cases|case [0-9]|first case|second case)\b', proof_text, re.IGNORECASE):
            techniques.append("case_analysis")
        
        # Look for direct proof
        if not techniques and re.search(r'\b(thus|therefore|hence)\b', proof_text, re.IGNORECASE):
            techniques.append("direct")
        
        # Default if no technique is identified
        if not techniques:
            techniques.append("unknown")
        
        return techniques
    
    def suggest_proof_approach(self, domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest a proof approach based on the domain information.
        
        Args:
            domain_info: The domain information
            
        Returns:
            A dictionary with approach suggestions
        """
        primary_domain = domain_info.get("primary_domain", "general")
        key_concepts = domain_info.get("key_concepts", [])
        
        # Domain-specific proof approaches
        approaches = []
        tactics = []
        libraries = []
        
        if primary_domain == "11":  # Number Theory
            approaches.append("Consider special cases (e.g., even vs. odd)")
            approaches.append("Look for divisibility patterns")
            tactics.append("lia (Linear Integer Arithmetic)")
            tactics.append("induction")
            libraries.append("Arith")
            libraries.append("ZArith")
        
        elif primary_domain in ["12", "13", "15", "16", "17", "18", "19", "20"]:  # Algebra
            approaches.append("Use algebraic properties and axioms")
            approaches.append("Apply homomorphism properties")
            tactics.append("ring")
            tactics.append("field")
            libraries.append("Algebra")
        
        elif primary_domain in ["26", "28", "30", "31", "32", "33", "34", "35"]:  # Analysis
            approaches.append("Use epsilon-delta definitions")
            approaches.append("Apply theorems about limits, continuity, etc.")
            tactics.append("auto with real")
            libraries.append("Reals")
        
        elif primary_domain in ["54", "55"]:  # Topology
            approaches.append("Use open/closed set properties")
            approaches.append("Apply continuity definitions")
            libraries.append("Topology")
        
        elif primary_domain in ["51", "52", "53"]:  # Geometry
            approaches.append("Use coordinate representations")
            approaches.append("Apply geometric theorems")
            tactics.append("auto with geo")
            libraries.append("Geometry")
        
        # Default approaches
        if not approaches:
            approaches.append("Start with a direct proof approach")
            approaches.append("Consider the definitions of all terms")
            tactics.append("auto")
            tactics.append("simpl")
        
        return {
            "suggested_approaches": approaches,
            "suggested_tactics": tactics,
            "suggested_libraries": libraries,
            "explanation": f"Based on the {MSC_CATEGORIES.get(primary_domain.split('-')[0], 'General Mathematics')} domain of this problem, focusing on concepts like {', '.join(key_concepts[:3])}."
        }

# Example usage
if __name__ == "__main__":
    detector = DomainDetector()
    
    # Test with a number theory example
    theorem = "For any natural number n, n² + n is even."
    proof = "Let n be a natural number. If n is even, then n² is even, and so n² + n is even. If n is odd, then n² is odd, and so n² + n is even. Therefore, in all cases, n² + n is even."
    
    domain_info = detector.detect_domain(theorem, proof)
    print(f"Domain: {domain_info['primary_domain']} ({domain_info['msc_name']})")
    print(f"Confidence: {domain_info['confidence']:.2f}")
    print(f"Key concepts: {domain_info['key_concepts']}")
    print(f"Proof techniques: {domain_info['proof_techniques']}")
    
    # Get suggestions
    suggestions = detector.suggest_proof_approach(domain_info)
    print("\nSuggested approaches:")
    for approach in suggestions['suggested_approaches']:
        print(f"- {approach}")
    
    print("\nSuggested tactics:")
    for tactic in suggestions['suggested_tactics']:
        print(f"- {tactic}")
    
    print(f"\nExplanation: {suggestions['explanation']}")