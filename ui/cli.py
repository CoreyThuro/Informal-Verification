"""
Command-line interface for the proof translation system.
Provides a CLI for translating informal proofs to formal proofs.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nlp.proof_parser import parse_math_proof
from nlp.domain_detector import detect_domain
from nlp.pattern_recognizer import recognize_pattern
from ir.proof_builder import build_proof_ir
from translation.strategy_selector import select_translation_strategy
from backends.backend_interface import BackendRegistry

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate informal mathematical proofs to formal proofs in Coq or Lean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ui.cli --text "Assume x is a natural number. Then x + x is even." --prover coq
  python -m ui.cli --file proof.txt --prover lean
  python -m ui.cli --text "Theorem: For any natural number n, n^2 >= n." --interactive --prover coq
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
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive session with the prover after translation"
    )
    parser.add_argument(
        "--no-verify", "-n",
        action="store_true",
        help="Skip verification of the proof"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--llm", "-l",
        action="store_true",
        help="Use LLM assistance for translation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    return parser.parse_args()

def read_input(args):
    """Read input from file or command line."""
    if args.file:
        try:
            with open(args.file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        return args.text

def print_debug_info(parsed_info, domain_info, pattern_info, strategy_info, args):
    """Print debug information."""
    print("\n=== Debug Information ===")
    
    # Print basic info
    print("\nInput Text:")
    print(f"  Theorem: {parsed_info['theorem_text']}")
    print(f"  Proof: {parsed_info['proof_text']}")
    
    # Print domain info
    print("\nDomain Detection:")
    print(f"  Primary Domain: {domain_info['primary_domain']}")
    print(f"  Confidence: {domain_info['confidence']:.2f}")
    print(f"  MSC Code: {domain_info['msc_code']}")
    print(f"  MSC Name: {domain_info['msc_name']}")
    
    # Print pattern info
    print("\nPattern Recognition:")
    print(f"  Primary Pattern: {pattern_info['primary_pattern']['name']}")
    print(f"  Confidence: {pattern_info['primary_pattern']['confidence']:.2f}")
    print(f"  Description: {pattern_info['primary_pattern']['description']}")
    
    # Print strategy info
    print("\nTranslation Strategy:")
    print(f"  Selected Strategy: {strategy_info['strategy'].value}")
    print(f"  Target Prover: {args.prover}")
    
    # Print structure info
    if args.verbose:
        print("\nProof Structure:")
        print(f"  Assumptions: {len(parsed_info['proof_structure']['assumptions'])}")
        print(f"  Conclusions: {len(parsed_info['proof_structure']['conclusions'])}")
        print(f"  Variables: {parsed_info['proof_structure']['variables']}")
        
        print("\nStrategy Configuration:")
        for key, value in strategy_info['config']['parameters'].items():
            print(f"  {key}: {value}")
    
    print("\n======================\n")

def process_proof(input_text, args):
    """Process the proof and generate formal output."""
    print("Processing proof...")
    
    # Parse the input
    parsed_info = parse_math_proof(input_text)
    
    if args.debug:
        print(f"Parsed {len(parsed_info['parsed_statements'])} statements.")
    
    # Extract domain information
    domain_info = detect_domain(
        parsed_info['theorem_text'],
        parsed_info['proof_text']
    )
    
    # Recognize pattern
    pattern_info = recognize_pattern(parsed_info['proof_text'])
    
    # Build intermediate representation
    proof_ir = build_proof_ir(
        parsed_statements=parsed_info['parsed_statements'],
        proof_structure=parsed_info['proof_structure'],
        original_theorem=parsed_info['theorem_text'],
        original_proof=parsed_info['proof_text']
    )
    
    # Select translation strategy
    strategy_info = select_translation_strategy(
        proof_ir=proof_ir,
        target_prover=args.prover,
        use_llm=args.llm
    )
    
    # Print debug info if requested
    if args.debug:
        print_debug_info(parsed_info, domain_info, pattern_info, strategy_info, args)
    
    # Get the appropriate backend
    try:
        backend = BackendRegistry.get_backend(args.prover)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    # Check if prover is installed
    if not backend.is_installed() and not args.no_verify:
        print(f"Warning: {args.prover} is not installed. Verification will be skipped.")
        args.no_verify = True
    
    # Translate to formal proof
    print(f"Translating to {args.prover}...")
    formal_proof = backend.translate(proof_ir)
    
    # Output the proof
    if args.output:
        with open(args.output, 'w') as f:
            f.write(formal_proof)
        print(f"Formal proof written to {args.output}")
    else:
        print("\n=== Formal Proof ===")
        print(formal_proof)
        print("===================\n")
    
    # Verify the proof if requested
    if not args.no_verify:
        print("Verifying proof...")
        success, error_message = backend.verify(formal_proof)
        
        if success:
            print("✅ Proof verified successfully!")
        else:
            print("❌ Proof verification failed:")
            print(error_message)
            
            # Process error feedback
            error_info = backend.process_feedback(error_message)
            
            if error_info.get("suggestion"):
                print(f"\nSuggestion: {error_info['suggestion']}")
    
    # Start interactive session if requested
    if args.interactive:
        temp_file = backend.interactive_session(formal_proof)
        print(f"Interactive session file: {temp_file}")
    
    return formal_proof

def main():
    """Main function for the CLI."""
    args = parse_arguments()
    input_text = read_input(args)
    
    try:
        formal_proof = process_proof(input_text, args)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())