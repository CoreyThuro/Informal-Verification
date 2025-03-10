"""
OpenAI client for the LLM interface.
Implements the LLM interface using OpenAI's API.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union
import time

from llm.llm_interface import LLMInterface, LLMFactory

class OpenAIClient(LLMInterface):
    """
    OpenAI API client implementing the LLM interface.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: Optional API key, uses environment variable if None
            model: The model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI Python library not installed. Install with 'pip install openai'.")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using OpenAI.
        
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
        
        # Make the API call with retry for rate limiting
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **params
                )
                
                # Extract and return the content
                return response.choices[0].message.content
            
            except Exception as e:
                if "rate limit" in str(e).lower() and retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise
    
    def generate_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON data from a prompt using OpenAI.
        
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
        Get embedding vector for text using OpenAI.
        
        Args:
            text: The text to embed
            **kwargs: Additional parameters for the model
            
        Returns:
            List of embedding values
        """
        # Default embedding model
        model = kwargs.get("model", "text-embedding-ada-002")
        
        # Make the API call
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        
        # Extract and return the embedding
        return response.data[0].embedding


# Register the interface
LLMFactory.register("openai", OpenAIClient)


# Utility functions for OpenAI

def get_openai_client(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> OpenAIClient:
    """
    Get an OpenAI client.
    
    Args:
        api_key: Optional API key, uses environment variable if None
        model: The model to use
        
    Returns:
        An OpenAI client
    """
    return OpenAIClient(api_key, model)

def translate_proof_with_openai(theorem: str, proof: str, target_prover: str, 
                               model: str = "gpt-3.5-turbo") -> str:
    """
    Translate a proof using OpenAI.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        target_prover: The target theorem prover
        model: The model to use
        
    Returns:
        The translated proof
    """
    from llm.llm_interface import translate_proof_with_llm
    
    client = get_openai_client(model=model)
    return translate_proof_with_llm(theorem, proof, target_prover, client)

def analyze_proof_with_openai(theorem: str, proof: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Analyze a proof using OpenAI.
    
    Args:
        theorem: The theorem statement
        proof: The proof text
        model: The model to use
        
    Returns:
        Dictionary with analysis information
    """
    from llm.llm_interface import analyze_proof_with_llm
    
    client = get_openai_client(model=model)
    return analyze_proof_with_llm(theorem, proof, client)

def verify_openai_setup():
    """
    Verify that the OpenAI client can be initialized with the API key.
    
    Returns:
        Tuple of (is_configured, message)
    """
    import os
    
    # Check if API key is in environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY environment variable not set"
    
    try:
        # Try to import OpenAI
        from openai import OpenAI
        
        # Try to initialize the client
        client = OpenAI(api_key=api_key)
        
        # If we got here, initialization was successful
        return True, f"OpenAI client initialized successfully. API key starts with: {api_key[:4]}..."
    except ImportError:
        return False, "OpenAI Python library not installed. Install with 'pip install openai'"
    except Exception as e:
        return False, f"Error initializing OpenAI client: {str(e)}"