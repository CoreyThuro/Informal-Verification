# ui/hybrid_cli.py
import argparse
import sys
import os
import logging
from typing import Dict, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation.hybrid_translator import HybridTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hybrid_cli")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate informal mathematical proofs to formal proofs using a hybrid approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ui.hybrid_cli --text "Assume x is a natural number. Then x + x is even." --prover coq
  python -m ui.hybrid_cli --file proof.txt --prover lean
"""
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        help="The informal proof text"
    )
    input_group.add_argument(
        "--file", "-f",
        help="File containing the informal proof text"
    )
    
    parser.add_argument(
        "--prover", "-p",
        choices=["coq", "lean"],
        default="coq",
        help="Target theorem prover (default: coq)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for the formal proof (default: print to stdout)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to a pre-trained model"
    )
    parser.add_argument(
        "--use-llm", "-l",
        action="store_true",
        help="Use LLM assistance for translation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def read_input(args):
    """Read input from file or command line."""
    if args.file:
        try:
            with open(args.file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        return args.text

def extract_theorem_and_proof(text: str) -> tuple:
    """
    Extract theorem and proof from text.
    
    Args:
        text: Input text containing theorem and proof
        
    Returns:
        Tuple of (theorem, proof)
    """
    # Look for keywords like "Theorem:" or "Proof:"
    import re
    
    # Try to split by "Proof:"
    proof_split = re.split(r'(?i)proof[:\.]', text, 1)
    
    if len(proof_split) > 1:
        # We found a proof marker
        theorem_part = proof_split[0].strip()
        proof_part = proof_split[1].strip()
        
        # If theorem part contains "Theorem:", extract from there
        theorem_split = re.split(r'(?i)theorem[:\.]', theorem_part, 1)
        if len(theorem_split) > 1:
            theorem_part = theorem_split[1].strip()
        
        return theorem_part, proof_part
    
    # If no proof marker, try to split by "Theorem:"
    theorem_split = re.split(r'(?i)theorem[:\.]', text, 1)
    if len(theorem_split) > 1:
        # Assume everything is the theorem
        return theorem_split[1].strip(), ""
    
    # If we can't split, return the whole text as both theorem and proof
    return text, text

def process_proof(input_text: str, args) -> Dict[str, Any]:
    """
    Process the proof and generate formal output.
    
    Args:
        input_text: The input text containing theorem and proof
        args: Command-line arguments
        
    Returns:
        Dictionary with results
    """
    logger.info("Processing proof...")
    
    # Extract theorem and proof from input text
    theorem_text, proof_text = extract_theorem_and_proof(input_text)
    
    # If we couldn't separate theorem and proof, use the whole text as proof
    if not theorem_text:
        theorem_text = proof_text
    
    if args.verbose:
        logger.info(f"Theorem: {theorem_text}")
        logger.info(f"Proof: {proof_text}")
    
    # Create translator
    translator = HybridTranslator(
        target_prover=args.prover,
        model_path=args.model,
        use_llm=args.use_llm
    )
    
    # Translate
    logger.info(f"Translating to {args.prover}...")
    result = translator.translate(theorem_text, proof_text)
    
    # Output the proof
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result["formal_proof"])
        logger.info(f"Formal proof written to {args.output}")
    else:
        print("\n=== Formal Proof ===")
        print(result["formal_proof"])
        print("===================\n")
    
    # Show verification result
    if result["verified"]:
        logger.info("✅ Proof verified successfully!")
    else:
        logger.info("❌ Proof verification failed:")
        if result["error_message"]:
            print(result["error_message"])
    
    # Show metadata
    if args.verbose:
        print("\n=== Metadata ===")
        print(f"Domain: {result['domain']}")
        print(f"Pattern: {result['pattern']}")
        print("===============\n")
    
    return result

def main():
    """Main function for the CLI."""
    args = parse_arguments()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_text = read_input(args)
    
    try:
        result = process_proof(input_text, args)
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())