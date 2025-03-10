#!/usr/bin/env python3
"""
Test script to verify that LLM integration is working.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_test")

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_openai_integration():
    """Test OpenAI integration."""
    
    from llm.openai_client import verify_openai_setup
    
    # Check for API key in environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables.")
        logger.info("Would you like to enter an API key for testing? (y/n): ")
        response = input().strip().lower()
        
        if response == 'y':
            logger.info("Enter your OpenAI API key: ")
            api_key = input().strip()
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("API key set for this session.")
        else:
            logger.info("Skipping OpenAI tests that require API key.")
            return False
    
    # Check setup
    is_configured, message = verify_openai_setup()
    logger.info(f"OpenAI setup: {message}")
    
    if not is_configured:
        logger.error("OpenAI is not properly configured. Exiting test.")
        return False
    
    # Try to create client
    try:
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        logger.info("Successfully created OpenAI client")
        
        # Try a simple query
        prompt = "Translate the following proof to Coq: 'Let x be a natural number. Then x + x is even.'"
        logger.info(f"Sending test prompt: {prompt}")
        
        response = client.generate_text(prompt, max_tokens=100)
        logger.info(f"Received response (truncated): {response[:100]}...")
        
        logger.info("OpenAI integration test successful!")
        return True
    except Exception as e:
        logger.error(f"Error using OpenAI client: {str(e)}")
        return False

def test_proof_translation_with_llm():
    """Test translating a proof with LLM assistance."""
    
    from nlp.proof_parser import parse_math_proof
    from translation.strategy_selector import get_optimal_strategy
    
    theorem = "For any natural number n, n + n is even."
    proof = "Let n be a natural number. Then n + n = 2n, which is even by definition."
    
    logger.info("Parsing proof...")
    parsed_info = parse_math_proof(proof)
    
    logger.info("Getting optimal strategy with LLM...")
    try:
        strategy_info = get_optimal_strategy(theorem, proof, "coq", use_llm=True)
        
        logger.info(f"Strategy: {strategy_info.get('strategy', 'unknown')}")
        logger.info(f"Strategy parameters: {strategy_info.get('config', {}).get('parameters', {})}")
        
        if strategy_info.get('strategy') == "LLM_ASSISTED":
            logger.info("✅ LLM assistance is being used!")
        else:
            logger.warning("⚠️ LLM assistance is NOT being used.")
        
        return True
    except Exception as e:
        logger.error(f"Error in translation with LLM: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_fallback_to_rule_based():
    """Test fallback to rule-based translation when LLM is not available."""
    
    from nlp.proof_parser import parse_math_proof
    from translation.strategy_selector import get_optimal_strategy
    
    # Temporarily ensure no API key is available
    original_key = os.environ.get("OPENAI_API_KEY")
    if original_key:
        os.environ.pop("OPENAI_API_KEY")
    
    theorem = "For any natural number n, n + n is even."
    proof = "Let n be a natural number. Then n + n = 2n, which is even by definition."
    
    logger.info("Testing fallback to rule-based translation...")
    try:
        strategy_info = get_optimal_strategy(theorem, proof, "coq", use_llm=True)
        
        logger.info(f"Strategy when API key missing: {strategy_info.get('strategy', 'unknown')}")
        
        if strategy_info.get('strategy') != "LLM_ASSISTED":
            logger.info("✅ Correctly fell back to rule-based translation!")
        else:
            logger.warning("⚠️ Using LLM_ASSISTED strategy even though API key is missing!")
        
        # Restore original key if there was one
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        
        return True
    except Exception as e:
        logger.error(f"Error in fallback test: {str(e)}")
        
        # Restore original key if there was one
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        
        return False

if __name__ == "__main__":
    logger.info("Testing OpenAI integration...")
    has_openai = test_openai_integration()
    
    if has_openai:
        logger.info("\nTesting proof translation with LLM...")
        test_proof_translation_with_llm()
    else:
        logger.info("\nSkipping LLM translation test due to missing OpenAI configuration.")
        
    logger.info("\nTesting fallback to rule-based translation...")
    test_fallback_to_rule_based()