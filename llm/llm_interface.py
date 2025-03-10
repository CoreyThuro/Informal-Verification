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

def translate_proof_with_llm(theorem: str, proof: str, target_prover: str, 
                            llm: Optional[LLMInterface] = None) -> str:
    """
    Translate a proof using an LLM.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        target_prover: The target theorem prover
        llm: Optional LLM interface, uses default if None
        
    Returns:
        The translated proof
    """
    if llm is None:
        llm = get_default_llm()
    
    prompt = DEFAULT_PROOF_TRANSLATION_PROMPT.format(
        theorem=theorem,
        proof=proof,
        prover=target_prover
    )
    
    return llm.generate_text(prompt, temperature=0.2, max_tokens=2000)

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