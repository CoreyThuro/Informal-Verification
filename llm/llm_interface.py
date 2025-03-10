"""
Interface for language model integration.
Defines a common interface for different LLM providers.
"""

from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Any, Union
import os
import json

class LLMInterface(ABC):
    """
    Abstract base class for language model interfaces.
    """
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated text
        """
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON data from a prompt.
        
        Args:
            prompt: The prompt text
            schema: The JSON schema to follow
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated JSON data
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: The text to embed
            **kwargs: Additional parameters for the model
            
        Returns:
            List of embedding values
        """
        pass

class LLMFactory:
    """
    Factory for creating LLM interfaces.
    """
    
    _interfaces = {}
    
    @classmethod
    def register(cls, name: str, interface_class):
        """
        Register an LLM interface.
        
        Args:
            name: The name to register the interface under
            interface_class: The interface class
        """
        cls._interfaces[name.lower()] = interface_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> LLMInterface:
        """
        Create an LLM interface.
        
        Args:
            name: The name of the interface to create
            **kwargs: Parameters for the interface
            
        Returns:
            An LLM interface instance
            
        Raises:
            ValueError: If the interface is not registered
        """
        name = name.lower()
        if name not in cls._interfaces:
            registered = ", ".join(cls._interfaces.keys())
            raise ValueError(f"Unknown LLM interface '{name}'. Registered interfaces: {registered}")
        
        return cls._interfaces[name](**kwargs)
    
    @classmethod
    def list_interfaces(cls) -> List[str]:
        """
        List all registered interfaces.
        
        Returns:
            List of interface names
        """
        return list(cls._interfaces.keys())


# Default prompt templates for common tasks
DEFAULT_PROOF_TRANSLATION_PROMPT = """
Translate the following informal mathematical proof to formal {prover} proof:

Theorem: {theorem}

Proof:
{proof}

Translate this to {prover} syntax following these guidelines:
- Use standard libraries where appropriate
- Make sure to define all variables and their types
- Structure the proof clearly with appropriate tactics
- Ensure the proof is complete and verifiable

{prover} Translation:
"""

DEFAULT_PROOF_ANALYSIS_PROMPT = """
Analyze the following mathematical proof:

Theorem: {theorem}

Proof:
{proof}

Please provide a detailed analysis including:
1. The mathematical domain (e.g., algebra, analysis, number theory)
2. The proof technique being used (e.g., induction, contradiction)
3. The key steps in the proof
4. Any potential gaps or assumptions
5. How this would be formalized in a theorem prover

Your analysis:
"""

DEFAULT_THEOREM_STRUCTURE_PROMPT = """
Extract the logical structure of the following mathematical theorem:

{theorem}

Please output a JSON object with the following structure:
{
    "statement_type": "universal/existential/implication/etc",
    "variables": [list of variables used],
    "premises": [list of premises/assumptions],
    "conclusion": "the conclusion statement",
    "quantifiers": [list of quantifiers and their variables],
    "domain": "likely mathematical domain"
}

JSON structure:
"""

# Utility functions

def get_default_llm() -> LLMInterface:
    """
    Get the default LLM interface based on environment.
    
    Returns:
        An LLM interface instance
    
    Raises:
        ValueError: If no default interface can be determined
    """
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return LLMFactory.create("openai")
        except (ValueError, ImportError):
            pass
    
    # Check for other providers
    for provider in ["anthropic", "local", "huggingface"]:
        try:
            return LLMFactory.create(provider)
        except (ValueError, ImportError):
            continue
    
    raise ValueError("No default LLM interface found. Please specify an interface explicitly.")

"""
LLM translation function with enhanced prompting.
"""

import re
import os
from typing import Dict, List, Any, Optional, Union

def get_example_translations(target_prover: str) -> str:
    """Get examples of translations for the specified prover."""
    
    if target_prover.lower() == "coq":
        return """
## Example 1: Evenness of sum

Theorem: For any natural number n, n + n is even.
Proof: Let n be a natural number. Then n + n = 2n, which is even by definition.

Coq translation:
```coq
Require Import Arith.
Require Import Lia.

Theorem n_plus_n_even: forall n: nat, exists k: nat, n + n = 2 * k.
Proof.
  intros n.
  exists n.
  ring.
Qed.
```

## Example 2: Irrationality of sqrt(2)

Theorem: The square root of 2 is irrational.
Proof: Suppose sqrt(2) is rational, so sqrt(2) = a/b for integers a,b with gcd(a,b)=1 and b≠0.
Then 2 = a²/b², so 2b² = a². Thus a² is even, so a is even. 
Let a = 2c. Then 2b² = 4c², so b² = 2c². Thus b is even.
But then a and b are both even, contradicting gcd(a,b)=1.

Coq translation:
```coq
Require Import Reals.
Require Import Lia.
Require Import ZArith.
Require Import Znumtheory.

Theorem sqrt2_irrational: ~ exists (p q : Z), q <> 0 /\ (Zgcd p q) = 1 /\ sqrt 2 = (IZR p / IZR q)%R.
Proof.
  intros [p [q [qNon0 [coprime sqrt2_rat]]]].
  assert (sqr_rational: 2 = (IZR (p * p) / IZR (q * q))%R).
  {
    replace (IZR (p * p)) with (IZR p * IZR p)%R by (apply IZR_mul).
    replace (IZR (q * q)) with (IZR q * IZR q)%R by (apply IZR_mul).
    rewrite <- sqrt2_rat.
    rewrite <- sqrt_def.
    - reflexivity.
    - apply Rlt_0_2.
  }
  
  (* Now derive 2 * q² = p² *)
  assert (2 * q * q = p * p).
  {
    apply eq_IZR.
    field_simplify in sqr_rational.
    field_simplify.
    rewrite sqr_rational.
    field.
    split; assumption.
  }
  
  (* p is even *)
  assert (even_p: exists k, p = 2 * k).
  {
    apply Zeven_2n.
    apply Zeven_mult_Zeven_r.
    exists (q * q).
    assumption.
  }
  destruct even_p as [k p_eq].
  
  (* Derive that q is even *)
  assert (2 * k * k = q * q).
  {
    rewrite p_eq in H.
    rewrite Z.mul_assoc in H.
    apply Zmult_reg_l in H.
    - assumption.
    - discriminate.
  }
  
  assert (even_q: exists j, q = 2 * j).
  {
    apply Zeven_2n.
    apply Zeven_mult_Zeven_r.
    exists k.
    assumption.
  }
  
  (* Contradiction: p and q cannot both be even if gcd(p,q)=1 *)
  destruct even_q as [j q_eq].
  assert (2 | Zgcd p q).
  {
    rewrite coprime.
    apply Zdivide_1.
  }
  assert (2 | p) by (exists k; rewrite p_eq; ring).
  assert (2 | q) by (exists j; rewrite q_eq; ring).
  assert (2 | Zgcd p q) by (apply Zgcd_is_gcd; assumption).
  contradiction.
Qed.
```

## Example 3: Transitivity of divisibility

Theorem: If a divides b and b divides c, then a divides c.
Proof: Assume a divides b and b divides c. Then there exist integers k and m such that b = ka and c = mb. Substituting, we get c = m(ka) = (mk)a. This shows that a divides c.

Coq translation:
```coq
Require Import ZArith.
Require Import Znumtheory.

Theorem div_trans: forall a b c : Z, (a | b) -> (b | c) -> (a | c).
Proof.
  intros a b c H1 H2.
  unfold Z.divide in *.
  destruct H1 as [k H1].
  destruct H2 as [m H2].
  exists (m * k).
  rewrite H2.
  rewrite H1.
  ring.
Qed.
```
"""
    elif target_prover.lower() == "lean":
        return """
## Example 1: Evenness of sum

Theorem: For any natural number n, n + n is even.
Proof: Let n be a natural number. Then n + n = 2n, which is even by definition.

Lean translation:
```lean
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic.Ring

theorem n_plus_n_even : ∀ n : ℕ, ∃ k : ℕ, n + n = 2 * k := by
  intro n
  use n
  ring
```

## Example 2: Irrationality of sqrt(2)

Theorem: The square root of 2 is irrational.
Proof: Suppose sqrt(2) is rational, so sqrt(2) = a/b for integers a,b with gcd(a,b)=1 and b≠0.
Then 2 = a²/b², so 2b² = a². Thus a² is even, so a is even. 
Let a = 2c. Then 2b² = 4c², so b² = 2c². Thus b is even.
But then a and b are both even, contradicting gcd(a,b)=1.

Lean translation:
```lean
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Int.GCD
import Mathlib.Tactic.Ring

theorem sqrt2_irrational : ¬(∃ (p q : ℤ), q ≠ 0 ∧ Int.gcd p q = 1 ∧ Real.sqrt 2 = p / q) := by
  -- Assume sqrt(2) is rational
  intro h
  rcases h with ⟨p, q, q_nonzero, coprime, sqrt2_eq⟩
  
  -- Square both sides to get 2 = (p/q)²
  have sqr_eq : 2 = p^2 / q^2 := by
    rw [sqrt2_eq]
    simp [Real.sqrt_sq, div_pow]
    apply Nat.cast_pos.mpr
    norm_num
  
  -- Derive 2q² = p²
  have eq1 : 2 * q^2 = p^2 := by
    field_simp at sqr_eq
    rw [mul_comm q^2 2]
    exact sqr_eq
  
  -- p is even
  have p_even : ∃ k, p = 2 * k := by
    apply Int.even_iff_two_dvd.mp
    apply Int.even_iff_two_dvd.mpr
    use q^2
    exact eq1
  
  -- Get the value k where p = 2k
  rcases p_even with ⟨k, p_eq⟩
  
  -- Derive that q is even
  have eq2 : 2 * k^2 = q^2 := by
    rw [p_eq] at eq1
    rw [pow_two, pow_two] at eq1
    ring at eq1
    have h1 : 2 * (2 * k)^2 = 2 * p^2 := by rw [p_eq]; ring
    have h2 : 2 * (2 * k)^2 = 2 * (2 * k * (2 * k)) := by ring
    have h3 : 2 * (2 * k * (2 * k)) = 8 * k^2 := by ring
    rw [h2, h3] at eq1
    have h4 : 8 * k^2 = 2 * q^2 := eq1
    apply mul_left_cancel₀ h4
    exact two_ne_zero
  
  -- q is even
  have q_even : ∃ j, q = 2 * j := by
    apply Int.even_iff_two_dvd.mp
    apply Int.even_iff_two_dvd.mpr
    use k^2
    exact eq2
  
  -- Get the value j where q = 2j
  rcases q_even with ⟨j, q_eq⟩
  
  -- Contradiction: p and q cannot both be even if gcd(p,q)=1
  have p_dvd_2 : 2 ∣ p := by use k; rw [p_eq]; ring
  have q_dvd_2 : 2 ∣ q := by use j; rw [q_eq]; ring
  
  have gcd_dvd_2 : 2 ∣ Int.gcd p q := by
    apply Int.gcd_dvd_gcd
    exact p_dvd_2
    exact q_dvd_2
  
  -- But gcd(p,q) = 1, so 2 | 1, which is impossible
  have absurd : 2 ∣ 1 := by
    rw [coprime] at gcd_dvd_2
    exact gcd_dvd_2
  
  exact Int.not_two_dvd_one absurd
```

## Example 3: Transitivity of divisibility

Theorem: If a divides b and b divides c, then a divides c.
Proof: Assume a divides b and b divides c. Then there exist integers k and m such that b = ka and c = mb. Substituting, we get c = m(ka) = (mk)a. This shows that a divides c.

Lean translation:
```lean
import Mathlib.Algebra.Divisibility.Basic

theorem div_trans {a b c : ℤ} : a ∣ b → b ∣ c → a ∣ c := by
  intro hab hbc
  rcases hab with ⟨k, rfl⟩
  rcases hbc with ⟨m, rfl⟩
  use m * k
  ring
```
"""
    else:
        return ""  # Default empty example set


def translate_proof_with_llm(theorem: str, proof: str, target_prover: str, 
                            llm=None) -> str:
    """
    Translate a proof using an LLM with enhanced prompting and examples.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        target_prover: The target theorem prover
        llm: Optional LLM interface, uses default if None
        
    Returns:
        The translated proof
    """
    if llm is None:
        # Import here to avoid circular imports
        from llm.llm_interface import get_default_llm
        llm = get_default_llm()
    
    # Get examples specific to the target prover
    examples = get_example_translations(target_prover)
    
    # Create the prompt
    prompt = f"""You are an expert in translating informal mathematical proofs into formal proofs in {target_prover}.

Translate the following mathematical theorem and its proof into a complete, syntactically correct {target_prover} program:

Theorem: {theorem}

Proof: {proof}

Your translation should:
1. Include all necessary library imports based on the mathematical domain
2. Define the theorem statement formally and precisely
3. Provide a complete proof using appropriate tactics
4. Be syntactically correct and verifiable in {target_prover}

{examples}

Now translate the given theorem and proof into {target_prover}:
"""
    
    # Generate the translation
    translation = llm.generate_text(prompt, temperature=0.3, max_tokens=2000)
    
    # Extract just the code block if present
    code_pattern = r'```(?:coq|lean)?\s*([\s\S]*?)```'
    code_match = re.search(code_pattern, translation)
    
    if code_match:
        return code_match.group(1).strip()
    else:
        return translation.strip()

def analyze_proof_with_llm(theorem: str, proof: str, 
                          llm: Optional[LLMInterface] = None) -> Dict[str, Any]:
    """
    Analyze a proof using an LLM.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        llm: Optional LLM interface, uses default if None
        
    Returns:
        Dictionary with analysis information
    """
    if llm is None:
        llm = get_default_llm()
    
    prompt = DEFAULT_PROOF_ANALYSIS_PROMPT.format(
        theorem=theorem,
        proof=proof
    )
    
    analysis_text = llm.generate_text(prompt, temperature=0.3, max_tokens=1000)
    
    # Try to extract structured information from the text
    try:
        return extract_structured_analysis(analysis_text)
    except:
        # Return as plain text if extraction fails
        return {"analysis": analysis_text}

def extract_structured_analysis(analysis_text: str) -> Dict[str, Any]:
    """
    Extract structured information from analysis text.
    
    Args:
        analysis_text: The text to analyze
        
    Returns:
        Dictionary with structured information
    """
    result = {
        "domain": None,
        "technique": None,
        "key_steps": [],
        "gaps": [],
        "formalization_notes": None
    }
    
    # Extract domain
    domain_match = re.search(r'domain.*?[:;]\s*([^\.]+)', analysis_text, re.IGNORECASE)
    if domain_match:
        result["domain"] = domain_match.group(1).strip()
    
    # Extract technique
    technique_match = re.search(r'technique.*?[:;]\s*([^\.]+)', analysis_text, re.IGNORECASE)
    if technique_match:
        result["technique"] = technique_match.group(1).strip()
    
    # Extract key steps
    if "key steps" in analysis_text.lower():
        steps_section = analysis_text.split("key steps", 1)[1].split("\n\n", 1)[0]
        steps = re.findall(r'\d+\.\s*([^\n]+)', steps_section)
        if steps:
            result["key_steps"] = [step.strip() for step in steps]
    
    # Extract gaps
    if "gaps" in analysis_text.lower():
        gaps_section = analysis_text.split("gaps", 1)[1].split("\n\n", 1)[0]
        gaps = re.findall(r'[-*]\s*([^\n]+)', gaps_section)
        if gaps:
            result["gaps"] = [gap.strip() for gap in gaps]
    
    # Extract formalization notes
    if "formalization" in analysis_text.lower():
        formalization_section = analysis_text.split("formalization", 1)[1].split("\n\n", 1)[0]
        result["formalization_notes"] = formalization_section.strip()
    
    return result