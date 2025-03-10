"""
Pattern recognizer for mathematical proofs.
Identifies common proof structures and patterns.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise ImportError("spaCy model 'en_core_web_sm' not found. Please install with 'python -m spacy download en_core_web_sm'")

class ProofPattern:
    """Class representing a proof pattern with its characteristics."""
    
    def __init__(self, name: str, description: str, keywords: List[str], structure: List[str] = None):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.structure = structure or []
    
    def __repr__(self):
        return f"ProofPattern({self.name})"

# Define common proof patterns
PROOF_PATTERNS = [
    ProofPattern(
        name="direct",
        description="Direct proof: starts with assumptions and derives the conclusion directly",
        keywords=["assume", "let", "given", "thus", "therefore", "hence", "so"],
        structure=["assumption", "steps", "conclusion"]
    ),
    
    ProofPattern(
        name="contradiction",
        description="Proof by contradiction: assumes the negation of the conclusion and derives a contradiction",
        keywords=["contradiction", "contrary", "false", "absurd", "suppose not", "assume not"],
        structure=["negation", "steps", "contradiction"]
    ),
    
    ProofPattern(
        name="contrapositive",
        description="Proof by contrapositive: proves 'if not B then not A' instead of 'if A then B'",
        keywords=["contrapositive", "converse", "not", "implies", "if not"],
        structure=["negation_of_conclusion", "steps", "negation_of_assumption"]
    ),
    
    ProofPattern(
        name="induction",
        description="Proof by induction: proves a base case and an inductive step",
        keywords=["induction", "base case", "inductive", "hypothesis", "step", "k", "k+1"],
        structure=["base_case", "inductive_step", "conclusion"]
    ),
    
    ProofPattern(
        name="case_analysis",
        description="Proof by cases: splits into different cases and proves each separately",
        keywords=["case", "cases", "first", "second", "either", "or", "otherwise"],
        structure=["case_1", "case_2", "case_n", "conclusion"]
    ),
    
    ProofPattern(
        name="existence",
        description="Existence proof: constructs or identifies an object with the required properties",
        keywords=["exists", "construct", "find", "example", "witness", "there is", "there exists"],
        structure=["construction", "verification", "conclusion"]
    ),
    
    ProofPattern(
        name="uniqueness",
        description="Uniqueness proof: assumes two objects satisfy the conditions and proves they are equal",
        keywords=["unique", "uniquely", "only", "at most one"],
        structure=["existence", "uniqueness_step", "conclusion"]
    ),
    
    ProofPattern(
        name="constructive",
        description="Constructive proof: explicitly constructs an object or algorithm",
        keywords=["construct", "algorithm", "procedure", "define", "construction"],
        structure=["construction", "verification", "conclusion"]
    ),
    
    ProofPattern(
        name="non_constructive",
        description="Non-constructive proof: proves existence without construction (e.g., using contradiction)",
        keywords=["exists", "contradiction", "suppose not", "cannot"],
        structure=["setup", "contradiction_argument", "conclusion"]
    ),
    
    ProofPattern(
        name="mathematical_induction",
        description="Proof by mathematical induction: base case + inductive step for integers",
        keywords=["induction", "base case", "n=0", "n=1", "k+1", "inductive", "hypothesis"],
        structure=["base_case", "inductive_hypothesis", "inductive_step", "conclusion"]
    ),
    
    ProofPattern(
        name="structural_induction",
        description="Proof by structural induction: induction on recursive structures",
        keywords=["induction", "structure", "recursive", "base case", "inductive"],
        structure=["base_case", "inductive_step", "conclusion"]
    ),
    
    ProofPattern(
        name="well_ordering",
        description="Proof using the well-ordering principle",
        keywords=["well-ordering", "minimal", "smallest", "least", "minimum"],
        structure=["assume_set", "minimal_element", "contradiction"]
    ),
    
    ProofPattern(
        name="pigeonhole",
        description="Proof using the pigeonhole principle",
        keywords=["pigeonhole", "boxes", "items", "more than", "at least", "mapping"],
        structure=["setup", "counting_argument", "conclusion"]
    ),
    
    ProofPattern(
        name="diagonalization",
        description="Proof using diagonalization",
        keywords=["diagonal", "diagonalization", "cantor", "enumeration", "list"],
        structure=["setup", "diagonal_argument", "conclusion"]
    ),
    
    ProofPattern(
        name="evenness",
        description="Proof that a number is even",
        keywords=["even", "divisible by 2", "2k"],
        structure=["assumption", "algebraic_manipulation", "conclusion"]
    ),
    
    ProofPattern(
        name="oddness",
        description="Proof that a number is odd",
        keywords=["odd", "2k+1", "not even"],
        structure=["assumption", "algebraic_manipulation", "conclusion"]
    )
]

class PatternRecognizer:
    """
    Recognizer for proof patterns in mathematical proofs.
    """
    
    def __init__(self):
        """Initialize the pattern recognizer."""
        self.patterns = {pattern.name: pattern for pattern in PROOF_PATTERNS}
    
    def recognize_pattern(self, proof_text: str) -> Dict[str, Any]:
        """
        Recognize the pattern in a proof.
        
        Args:
            proof_text: The proof text
            
        Returns:
            A dictionary with pattern information
        """
        # Convert to lowercase for case-insensitive matching
        proof_lower = proof_text.lower()
        
        # Score each pattern based on keyword matches
        pattern_scores = {}
        for name, pattern in self.patterns.items():
            score = self._score_pattern(proof_lower, pattern)
            pattern_scores[name] = score
        
        # Get the pattern with the highest score
        if pattern_scores:
            best_pattern_name = max(pattern_scores.items(), key=lambda x: x[1])[0]
            best_pattern = self.patterns[best_pattern_name]
            best_score = pattern_scores[best_pattern_name]
            
            # Calculate confidence (normalize the score)
            total_score = sum(pattern_scores.values())
            confidence = best_score / total_score if total_score > 0 else 0.0
            
            # Extract substructures if possible
            substructures = self._extract_substructures(proof_text, best_pattern)
            
            return {
                "pattern_name": best_pattern_name,
                "description": best_pattern.description,
                "confidence": confidence,
                "all_scores": pattern_scores,
                "substructures": substructures
            }
        else:
            return {
                "pattern_name": "unknown",
                "description": "No recognized pattern",
                "confidence": 0.0,
                "all_scores": {},
                "substructures": {}
            }
    
    def _score_pattern(self, proof_text: str, pattern: ProofPattern) -> float:
        """
        Score how well a proof matches a pattern based on keywords.
        
        Args:
            proof_text: The proof text (lowercase)
            pattern: The pattern to score
            
        Returns:
            A score representing the match quality
        """
        score = 0.0
        
        # Check for presence of keywords
        for keyword in pattern.keywords:
            if keyword in proof_text:
                # Count occurrences
                count = proof_text.count(keyword)
                score += count * 1.0  # Each occurrence adds to the score
        
        # Check for structural elements
        for structure_element in pattern.structure:
            # Look for indications of this structure element
            if self._has_structure_element(proof_text, structure_element):
                score += 2.0  # Higher weight for structural matches
        
        return score
    
    def _has_structure_element(self, proof_text: str, structure_element: str) -> bool:
        """
        Check if a proof text contains indicators of a specific structural element.
        
        Args:
            proof_text: The proof text (lowercase)
            structure_element: The structural element to check for
            
        Returns:
            True if the element is present, False otherwise
        """
        # Define indicators for common structural elements
        structure_indicators = {
            "assumption": ["assume", "let", "suppose", "given", "for any", "for all"],
            "base_case": ["base case", "n=0", "n=1", "initial", "first"],
            "inductive_step": ["inductive step", "inductive hypothesis", "assume for k", "holds for k+1"],
            "conclusion": ["therefore", "thus", "hence", "so", "we proved", "q.e.d", "as required"],
            "contradiction": ["contradiction", "absurd", "false", "impossible"],
            "case_1": ["case 1", "first case", "if", "when"],
            "case_2": ["case 2", "second case", "else", "otherwise"],
            "case_n": ["case n", "case i", "general case"],
            "construction": ["construct", "define", "let", "set"],
            "verification": ["verify", "check", "confirm", "satisfies"],
            "uniqueness_step": ["unique", "only one", "any two", "suppose there are two"],
            "negation": ["not", "negation", "assume the contrary", "suppose not"],
            "negation_of_assumption": ["not", "negation", "contrary"],
            "negation_of_conclusion": ["not", "negation", "contrary"],
            "steps": [".", ";"],  # Any sentence separator indicates steps
            "algebraic_manipulation": ["=", "+", "-", "*", "/", "substitute", "simplify"]
        }
        
        # Check if any indicators for this element are in the text
        if structure_element in structure_indicators:
            for indicator in structure_indicators[structure_element]:
                if indicator in proof_text:
                    return True
        
        return False
    
    def _extract_substructures(self, proof_text: str, pattern: ProofPattern) -> Dict[str, str]:
        """
        Extract substructures from a proof based on a pattern.
        
        Args:
            proof_text: The proof text
            pattern: The recognized pattern
            
        Returns:
            A dictionary mapping structure names to text excerpts
        """
        substructures = {}
        sentences = [s.strip() for s in re.split(r'[.;]', proof_text) if s.strip()]
        
        # Handle specific patterns
        if pattern.name == "induction" or pattern.name == "mathematical_induction":
            # Look for base case
            base_case_indices = []
            for i, sentence in enumerate(sentences):
                if re.search(r'base case|n\s*=\s*0|n\s*=\s*1|initial', sentence, re.IGNORECASE):
                    base_case_indices.append(i)
            
            # Look for inductive step
            inductive_step_indices = []
            for i, sentence in enumerate(sentences):
                if re.search(r'inductive|hypothesis|assume\s+for\s+k|k\s*\+\s*1', sentence, re.IGNORECASE):
                    inductive_step_indices.append(i)
            
            # Extract substructures if found
            if base_case_indices:
                start = base_case_indices[0]
                end = inductive_step_indices[0] if inductive_step_indices else len(sentences)
                substructures["base_case"] = ". ".join(sentences[start:end])
            
            if inductive_step_indices:
                start = inductive_step_indices[0]
                end = len(sentences)
                substructures["inductive_step"] = ". ".join(sentences[start:end])
        
        elif pattern.name == "contradiction":
            # Look for negation/assumption
            negation_indices = []
            for i, sentence in enumerate(sentences):
                if re.search(r'assume|suppose|negation|contrary|not', sentence, re.IGNORECASE):
                    negation_indices.append(i)
            
            # Look for contradiction
            contradiction_indices = []
            for i, sentence in enumerate(sentences):
                if re.search(r'contradiction|absurd|false|impossible', sentence, re.IGNORECASE):
                    contradiction_indices.append(i)
            
            # Extract substructures if found
            if negation_indices:
                start = negation_indices[0]
                end = contradiction_indices[0] if contradiction_indices else len(sentences)
                substructures["negation"] = ". ".join(sentences[start:end])
            
            if contradiction_indices:
                start = contradiction_indices[0]
                end = len(sentences)
                substructures["contradiction"] = ". ".join(sentences[start:end])
        
        elif pattern.name == "case_analysis":
            # Find all cases
            case_markers = [
                (r'case\s+1|first\s+case', "case_1"),
                (r'case\s+2|second\s+case', "case_2"),
                (r'case\s+3|third\s+case', "case_3"),
                (r'case\s+4|fourth\s+case', "case_4"),
                (r'case\s+n|case\s+i|general\s+case', "case_n")
            ]
            
            case_indices = []
            for i, sentence in enumerate(sentences):
                for pattern_regex, case_name in case_markers:
                    if re.search(pattern_regex, sentence, re.IGNORECASE):
                        case_indices.append((i, case_name))
            
            # Extract each case
            for i, (start_idx, case_name) in enumerate(case_indices):
                end_idx = case_indices[i+1][0] if i+1 < len(case_indices) else len(sentences)
                substructures[case_name] = ". ".join(sentences[start_idx:end_idx])
        
        # Default: extract assumption and conclusion
        if not substructures:
            # Assumption is typically at the beginning
            assumption_idx = -1
            for i, sentence in enumerate(sentences):
                if re.search(r'assume|let|suppose|given', sentence, re.IGNORECASE):
                    assumption_idx = i
                    break
            
            # Conclusion is typically at the end
            conclusion_idx = -1
            for i in range(len(sentences)-1, -1, -1):
                if re.search(r'therefore|thus|hence|proved|conclude', sentences[i], re.IGNORECASE):
                    conclusion_idx = i
                    break
            
            if assumption_idx >= 0:
                substructures["assumption"] = sentences[assumption_idx]
            
            if conclusion_idx >= 0:
                substructures["conclusion"] = sentences[conclusion_idx]
            
            # Middle steps
            if assumption_idx >= 0 and conclusion_idx >= 0 and conclusion_idx > assumption_idx + 1:
                substructures["steps"] = ". ".join(sentences[assumption_idx+1:conclusion_idx])
        
        return substructures
    
    def suggest_proof_strategy(self, theorem_text: str, domain_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest a proof strategy based on the theorem and domain.
        
        Args:
            theorem_text: The theorem text
            domain_info: Optional domain information
            
        Returns:
            A dictionary with strategy suggestions
        """
        # Parse the theorem
        doc = nlp(theorem_text)
        
        # Detect common forms that suggest specific strategies
        strategies = []
        explanations = []
        
        # Check for keywords that suggest proof technique
        if re.search(r'\b(every|all|any|for\s+all)\b', theorem_text, re.IGNORECASE):
            strategies.append("induction")
            explanations.append("The theorem involves a universal quantifier, suggesting induction might be appropriate.")
        
        if re.search(r'\b(exists|there\s+is|there\s+exists|some)\b', theorem_text, re.IGNORECASE):
            strategies.append("existence_proof")
            explanations.append("The theorem claims existence, suggesting a constructive approach.")
        
        if re.search(r'\b(unique|uniquely|only|at\s+most\s+one)\b', theorem_text, re.IGNORECASE):
            strategies.append("uniqueness_proof")
            explanations.append("The theorem involves uniqueness, suggesting a uniqueness proof.")
        
        if re.search(r'\b(if|imply|implies|when|whenever)\b', theorem_text, re.IGNORECASE):
            # Check if contrapositive or contradiction might be better
            if re.search(r'\b(not|no|never|impossible)\b', theorem_text, re.IGNORECASE):
                strategies.append("contrapositive")
                explanations.append("The theorem has a negative conclusion, suggesting contrapositive might work well.")
            else:
                strategies.append("direct")
                explanations.append("The theorem has an if-then structure, suggesting a direct proof.")
        
        if re.search(r'\b(even|odd|divisible|multiple|factor)\b', theorem_text, re.IGNORECASE):
            strategies.append("case_analysis")
            explanations.append("The theorem involves number properties, suggesting case analysis (e.g., even vs. odd).")
        
        if re.search(r'\b(prime|composite|gcd|lcm)\b', theorem_text, re.IGNORECASE):
            strategies.append("number_theory")
            explanations.append("The theorem involves number theory concepts.")
        
        # Use domain info if available
        if domain_info:
            primary_domain = domain_info.get("primary_domain", "")
            if primary_domain in ["11"]:  # Number Theory
                strategies.append("induction")
                explanations.append("Number theory problems often benefit from induction.")
                
                strategies.append("case_analysis")
                explanations.append("Number theory often requires case analysis (e.g., by congruence).")
            
            if primary_domain in ["26", "28", "30"]:  # Analysis
                strategies.append("epsilon_delta")
                explanations.append("Analysis problems often require epsilon-delta proofs.")
                
                strategies.append("contradiction")
                explanations.append("Analysis often uses proof by contradiction.")
        
        # Default strategy if none suggested
        if not strategies:
            strategies.append("direct")
            explanations.append("Start with a direct approach, working from the assumptions to the conclusion.")
        
        return {
            "suggested_strategies": strategies,
            "explanations": explanations,
            "theorem_structure": self._analyze_theorem_structure(theorem_text)
        }
    
    def _analyze_theorem_structure(self, theorem_text: str) -> Dict[str, Any]:
        """
        Analyze the structure of a theorem.
        
        Args:
            theorem_text: The theorem text
            
        Returns:
            A dictionary with structure information
        """
        # Simplistic analysis for now
        structure = {
            "has_universal_quantifier": bool(re.search(r'\b(every|all|any|for\s+all)\b', theorem_text, re.IGNORECASE)),
            "has_existential_quantifier": bool(re.search(r'\b(exists|there\s+is|there\s+exists|some)\b', theorem_text, re.IGNORECASE)),
            "has_uniqueness": bool(re.search(r'\b(unique|uniquely|only|at\s+most\s+one)\b', theorem_text, re.IGNORECASE)),
            "has_implication": bool(re.search(r'\b(if|imply|implies|when|whenever|then)\b', theorem_text, re.IGNORECASE)),
            "has_negation": bool(re.search(r'\b(not|no|never|impossible)\b', theorem_text, re.IGNORECASE)),
            "has_condition": bool(re.search(r'\b(if|when|whenever|assume|given)\b', theorem_text, re.IGNORECASE))
        }
        
        # Determine the overall structure
        if structure["has_implication"]:
            # Extract condition and conclusion
            match = re.search(r'if\s+(.*?)\s+then\s+(.*?)(?:$|\.)', theorem_text, re.IGNORECASE | re.DOTALL)
            if match:
                structure["condition"] = match.group(1).strip()
                structure["conclusion"] = match.group(2).strip()
                structure["form"] = "if_then"
            else:
                structure["form"] = "implication"
        elif structure["has_universal_quantifier"] and not structure["has_existential_quantifier"]:
            structure["form"] = "universal"
        elif structure["has_existential_quantifier"] and not structure["has_universal_quantifier"]:
            structure["form"] = "existential"
        elif structure["has_existential_quantifier"] and structure["has_uniqueness"]:
            structure["form"] = "unique_existence"
        else:
            structure["form"] = "generic"
        
        return structure

# Example usage
if __name__ == "__main__":
    recognizer = PatternRecognizer()
    
    # Test with an induction proof
    proof = """Proof by induction. 
    Base case: For n=1, we have 1 = 1(1+1)/2 = 1.
    Inductive step: Assume the formula holds for some k, i.e., 1 + 2 + ... + k = k(k+1)/2.
    Then for k+1, we have 1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.
    Therefore, the formula holds for all positive integers n."""
    
    pattern_info = recognizer.recognize_pattern(proof)
    print(f"Pattern: {pattern_info['pattern_name']}")
    print(f"Description: {pattern_info['description']}")
    print(f"Confidence: {pattern_info['confidence']:.2f}")
    
    if pattern_info['substructures']:
        print("\nSubstructures:")
        for name, text in pattern_info['substructures'].items():
            print(f"- {name}: {text}")
    
    # Test the strategy suggestion
    theorem = "For every positive integer n, the sum of the first n positive integers is n(n+1)/2."
    strategy_info = recognizer.suggest_proof_strategy(theorem)
    
    print("\nSuggested strategies:")
    for i, strategy in enumerate(strategy_info['suggested_strategies']):
        print(f"- {strategy}: {strategy_info['explanations'][i]}")
    
    print("\nTheorem structure:")
    for key, value in strategy_info['theorem_structure'].items():
        if not key.startswith("has_"):
            print(f"- {key}: {value}")