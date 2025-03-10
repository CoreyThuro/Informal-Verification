"""
Local LLM client for the LLM interface.
Implements the LLM interface using locally hosted models.
"""

import os
import json
import re
import requests
from typing import Dict, List, Optional, Any, Union

from llm.llm_interface import LLMInterface, LLMFactory

class LocalLLMClient(LLMInterface):
    """
    Client for locally hosted LLMs implementing the LLM interface.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000/v1", model: str = "default"):
        """
        Initialize the local LLM client.
        
        Args:
            api_url: The URL of the local LLM API
            model: The model to use
        """
        self.api_url = api_url
        self.model = model
        
        # Validate the API URL is accessible
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """
        Validate that the API URL is accessible.
        
        Raises:
            ConnectionError: If the API URL is not accessible
        """
        try:
            # Try to access the models endpoint
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Could not connect to local LLM API at {self.api_url}: {str(e)}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using a local LLM.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated text
        """
        # Default parameters
        params = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Update with provided parameters
        params.update(kwargs)
        
        # Extract the model and remove from params to avoid duplication
        model = params.pop("model", self.model)
        
        # Create the message
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided
        if "system_message" in params:
            messages.insert(0, {"role": "system", "content": params.pop("system_message")})
        
        # Make the API call
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    **params
                },
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Return the content of the first choice
            return response_data["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise RuntimeError(f"Error generating text with local LLM: {str(e)}")
    
    def generate_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON data from a prompt using a local LLM.
        
        Args:
            prompt: The prompt text
            schema: The JSON schema to follow
            **kwargs: Additional parameters for the model
            
        Returns:
            The generated JSON data
        """
        # Add schema to prompt
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nThe response should follow this JSON schema:\n{schema_str}\nJSON response:"
        
        # Add system message for JSON output
        kwargs["system_message"] = "You are a helpful assistant that returns valid JSON following the specified schema."
        
        # Generate response
        response_text = self.generate_text(full_prompt, **kwargs)
        
        # Extract JSON from response
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```|({[\s\S]*})'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            # Use the match from the first capturing group if available, otherwise the second
            json_str = json_match.group(1) or json_match.group(2)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found in response, try to parse the whole response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Return a minimal valid response if parsing fails
            return {"error": "Failed to parse JSON", "raw_response": response_text}
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Get embedding vector for text using a local embedding model.
        
        Args:
            text: The text to embed
            **kwargs: Additional parameters for the model
            
        Returns:
            List of embedding values
        """
        # Default embedding model
        model = kwargs.get("model", "embedding-model")
        
        # Make the API call
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Return the embedding of the first data item
            return response_data["data"][0]["embedding"]
            
        except Exception as e:
            raise RuntimeError(f"Error getting embedding with local LLM: {str(e)}")


# Register the interface
LLMFactory.register("local", LocalLLMClient)


# Utility functions for local LLMs

def get_local_llm_client(api_url: str = "http://localhost:8000/v1", 
                        model: str = "default") -> LocalLLMClient:
    """
    Get a local LLM client.
    
    Args:
        api_url: The URL of the local LLM API
        model: The model to use
        
    Returns:
        A local LLM client
    """
    return LocalLLMClient(api_url, model)

def translate_proof_with_local_llm(theorem: str, proof: str, target_prover: str, 
                                  api_url: str = "http://localhost:8000/v1",
                                  model: str = "default") -> str:
    """
    Translate a proof using a local LLM.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        target_prover: The target theorem prover
        api_url: The URL of the local LLM API
        model: The model to use
        
    Returns:
        The translated proof
    """
    from llm.llm_interface import translate_proof_with_llm
    
    client = get_local_llm_client(api_url, model)
    return translate_proof_with_llm(theorem, proof, target_prover, client)

def analyze_proof_with_local_llm(theorem: str, proof: str, 
                                api_url: str = "http://localhost:8000/v1",
                                model: str = "default") -> Dict[str, Any]:
    """
    Analyze a proof using a local LLM.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        api_url: The URL of the local LLM API
        model: The model to use
        
    Returns:
        Dictionary with analysis information
    """
    from llm.llm_interface import analyze_proof_with_llm
    
    client = get_local_llm_client(api_url, model)
    return analyze_proof_with_llm(theorem, proof, client)