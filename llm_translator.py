import re
import os
from typing import Dict, List, Tuple, Any

# Alternative: Use a locally running LLM via API
LLM_API_URL = "http://localhost:8000/v1/chat/completions"  # Updated for chat completions

class LLMTranslator:
    def __init__(self, model_name="gpt-4", use_local=False, few_shot_examples=None):
        """
        Initialize the LLM translator
        
        Args:
            model_name: The name of the model to use
            use_local: Whether to use a local LLM API instead of OpenAI
            few_shot_examples: Optional list of example translations for few-shot learning
        """
        self.model_name = model_name
        self.use_local = use_local
        
        # Default few-shot examples if none provided
        self.few_shot_examples = few_shot_examples or [
    {
        "informal": "Assume x is a natural number. Then x + x is even.",
        "coq": """Require Import Arith.
        Require Import ArithRing.

        Theorem example: forall x: nat, exists k: nat, x + x = 2 * k.
        Proof.
        intros x.
        exists x.
        simpl.
        reflexivity.
        Qed."""
            },
            {
                "informal": "If n is an even natural number, then n*n is even.",
                "coq": """Require Import Arith.
        Require Import ArithRing.

        Theorem square_even: forall n: nat, (exists k: nat, n = 2 * k) -> (exists m: nat, n * n = 2 * m).
        Proof.
        intros n [k Hk].
        exists (k * n).
        rewrite Hk.
        rewrite mult_assoc.
        simpl.
        reflexivity.
        Qed."""
            },
            {
                "informal": "Let n be an integer. If n is odd, then n^2 is odd.",
                "coq": """Require Import ZArith.
        Open Scope Z_scope.
        Require Import ZArithRing.

        Theorem odd_square: forall n: Z, (exists k: Z, n = 2 * k + 1) -> (exists j: Z, n^2 = 2 * j + 1).
        Proof.
        intros n [k H].
        exists (2 * k * k + 2 * k).
        rewrite H.
        ring.
        Qed."""
            },
            {
                "informal": "The sum of two even integers is even.",
                "coq": """Require Import ZArith.
        Open Scope Z_scope.
        Require Import ZArithRing.

        Theorem sum_even: forall a b: Z, (exists k: Z, a = 2 * k) -> (exists m: Z, b = 2 * m) -> (exists n: Z, a + b = 2 * n).
        Proof.
        intros a b [k Hk] [m Hm].
        exists (k + m).
        rewrite Hk, Hm.
        ring.
        Qed."""
            }
        ]
    
    def _build_prompt(self, informal_proof_text: str) -> str:
        """Build a prompt for the LLM with few-shot examples"""
        prompt = "Translate the following informal mathematical proofs to formal Coq proofs.\n\n"
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            prompt += f"Informal: {example['informal']}\n"
            prompt += f"Coq:\n{example['coq']}\n\n"
        
        # Add the target proof
        prompt += f"Informal: {informal_proof_text}\n"
        prompt += "Coq:\n"
        
        return prompt
    
    def _build_feedback_prompt(self, informal_proof_text: str) -> str:
        """Build a prompt to ask for mathematical feedback"""
        prompt = f"""Analyze the following informal mathematical proof for any logical or mathematical errors. 
If the proof is mathematically correct, respond with "PROOF_CORRECT". 
If there are errors, provide a brief explanation of the mathematical issues:

{informal_proof_text}

Mathematical feedback:"""
        
        return prompt
    
    def translate_with_openai(self, informal_proof_text: str) -> str:
        """Use OpenAI's API to translate the informal proof using the chat interface with v1.0.0+ API"""
        try:
            # Importing here to avoid dependency issues if the library is not installed
            from openai import OpenAI
            
            # Create a client using the API key from environment variable
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = self._build_prompt(informal_proof_text)
            
            # Make the API call with the new client-based format
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a mathematical proof translator that converts informal proofs to formal Coq proofs. Translate exactly what is given even if it contains mathematical errors. Do not include any explanations or corrections - output only valid Coq code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3  # Lower temperature for more deterministic outputs
            )
            
            # Extract the content from the new response format
            return response.choices[0].message.content.strip()
        except ImportError:
            return "Error: OpenAI library not installed or incompatible version. Install with 'pip install openai>=1.0.0'."
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return f"Error: {e}"
    
    def get_mathematical_feedback(self, informal_proof_text: str) -> str:
        """Get feedback on the mathematical correctness of a proof"""
        try:
            # Importing here to avoid dependency issues if the library is not installed
            from openai import OpenAI
            
            # Create a client using the API key from environment variable
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = self._build_feedback_prompt(informal_proof_text)
            
            # Make the API call with the new client-based format
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a mathematical assistant that analyzes proofs for logical and mathematical correctness."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.3
            )
            
            feedback = response.choices[0].message.content.strip()
            
            # If the feedback is "PROOF_CORRECT", return None to indicate no issues
            if feedback == "PROOF_CORRECT":
                return None
                
            return feedback
            
        except ImportError:
            return "Error: OpenAI library not installed."
        except Exception as e:
            print(f"Error getting mathematical feedback: {e}")
            return None
    
    def translate_with_local_llm(self, informal_proof_text: str) -> str:
        """Use a local LLM API to translate the informal proof using the chat interface"""
        import requests
        
        prompt = self._build_prompt(informal_proof_text)
        
        try:
            # Use the chat completions API format
            response = requests.post(
                LLM_API_URL,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a mathematical proof translator that converts informal proofs to formal Coq proofs. Translate exactly what is given even if it contains mathematical errors. Do not include any explanations or corrections - output only valid Coq code."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            print(f"Error using local LLM API: {e}")
            return f"Error: {e}"
    
    def get_mathematical_feedback_local(self, informal_proof_text: str) -> str:
        """Get feedback on the mathematical correctness of a proof using a local LLM"""
        import requests
        
        prompt = self._build_feedback_prompt(informal_proof_text)
        
        try:
            response = requests.post(
                LLM_API_URL,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a mathematical assistant that analyzes proofs for logical and mathematical correctness."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                feedback = response.json()["choices"][0]["message"]["content"].strip()
                
                # If the feedback is "PROOF_CORRECT", return None to indicate no issues
                if feedback == "PROOF_CORRECT":
                    return None
                    
                return feedback
            else:
                return None
        except Exception as e:
            print(f"Error getting mathematical feedback: {e}")
            return None
    
    def translate(self, informal_proof_text: str) -> str:
        """Translate an informal proof to Coq using an LLM"""
        if self.use_local:
            return self.translate_with_local_llm(informal_proof_text)
        else:
            return self.translate_with_openai(informal_proof_text)
    
    def get_feedback(self, informal_proof_text: str) -> str:
        """Get mathematical feedback for a proof"""
        if self.use_local:
            return self.get_mathematical_feedback_local(informal_proof_text)
        else:
            return self.get_mathematical_feedback(informal_proof_text)
    
    def verify_translation(self, coq_code: str) -> Tuple[bool, str]:
        """Verify that the generated Coq code is syntactically valid"""
        # This would call your existing Coq verification code
        # For now, we'll just check some basic patterns
        
        required_elements = ["Theorem", "Proof", "Qed"]
        for element in required_elements:
            if element not in coq_code:
                return False, f"Missing required element: {element}"
        
        return True, "Translation appears valid"
    
    def refine_translation(self, informal_proof_text: str, coq_code: str, error_message: str) -> str:
        """Refine a translation based on error feedback"""
        prompt = f"""The following Coq translation for an informal proof has errors:

Informal Proof:
{informal_proof_text}

Initial Coq Translation:
{coq_code}

Error Message:
{error_message}

Please fix the Coq translation to address these errors. Translate exactly what is given even if it contains mathematical errors. 
Output only valid Coq code without explanations or corrections:
"""
        
        if self.use_local:
            # Use local LLM to refine with chat API
            import requests
            
            try:
                response = requests.post(
                    LLM_API_URL,
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a mathematical proof translator that corrects and refines Coq proofs. Output only valid Coq code without explanations."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.3
                    }
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                else:
                    return coq_code  # Return original if refinement fails
            except Exception as e:
                print(f"Error refining with local LLM: {e}")
                return coq_code
        else:
            # Use OpenAI to refine with chat API (v1.0.0+)
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a mathematical proof translator that corrects and refines Coq proofs. Output only valid Coq code without explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
            except ImportError:
                print("Error: OpenAI library not installed or incompatible version.")
                return coq_code
            except Exception as e:
                print(f"Error refining with OpenAI: {e}")
                return coq_code

# Function to integrate with your existing pipeline
def translate_to_coq_with_llm(parsed_data, llm_translator=None):
    """Use LLM to translate parsed proof to Coq and provide mathematical feedback"""
    parsed_statements, proof_structure = parsed_data
    
    # Reconstruct the original proof text from parsed statements
    original_proof = " ".join([
        " ".join([token[0] for token in statement]) 
        for statement in parsed_statements
    ])
    
    # Initialize the translator if not provided
    if llm_translator is None:
        llm_translator = LLMTranslator()
    
    # Get mathematical feedback first
    print("Checking mathematical correctness...")
    feedback = llm_translator.get_feedback(original_proof)
    
    if feedback:
        print("\n🔍 Mathematical Feedback:")
        print(feedback)
        print("\nProceeding with translation regardless of mathematical correctness...\n")
    
    # Get the translation
    print("Translating to Coq...")
    coq_code = llm_translator.translate(original_proof)
    
    # Verify the translation
    is_valid, error_message = llm_translator.verify_translation(coq_code)
    
    # If not valid, try to refine it
    if not is_valid:
        print(f"Initial translation had issues: {error_message}")
        coq_code = llm_translator.refine_translation(original_proof, coq_code, error_message)
    
    # Add any missing imports or scope directives
    from coq_imports import add_required_imports
    coq_code = add_required_imports(coq_code)
    
    print("Generated Coq Code:")
    print(coq_code)
    
    return coq_code