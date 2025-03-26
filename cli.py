"""
Enhanced command-line interface for proof translation.
"""

import argparse
import sys
import os

from translator import ProofTranslator

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate informal mathematical proofs to Coq."
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--theorem", "-t",
        help="The theorem statement"
    )
    input_group.add_argument(
        "--file", "-f",
        help="File containing the theorem and proof (separate with 'Proof:')"
    )
    
    parser.add_argument(
        "--proof", "-p",
        help="The proof text (required if --theorem is used)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for the formal proof (default: stdout)"
    )
    
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the generated proof"
    )
    
    args = parser.parse_args()
    
    # Check arguments
    if args.theorem and not args.proof:
        print("Error: --proof is required when using --theorem")
        return 1
    
    # Read from file if specified
    if args.file:
        try:
            with open(args.file, 'r') as f:
                content = f.read()
                
            # Split into theorem and proof
            parts = content.split("Proof:", 1)
            if len(parts) < 2:
                print("Error: File must contain 'Proof:' separator")
                return 1
                
            theorem_text = parts[0].strip()
            proof_text = parts[1].strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    else:
        theorem_text = args.theorem
        proof_text = args.proof
    
    # Translate the proof
    translator = ProofTranslator()
    result = translator.translate(theorem_text, proof_text)
    
    # Output the result
    if args.output:
        with open(args.output, "w") as f:
            f.write(result["formal_proof"])
        print(f"Proof written to {args.output}")
        
        # Print verification result if requested
        if args.verify or result["verified"]:
            print(f"Verification: {'Success' if result['verified'] else 'Failed'}")
            if not result["verified"] and result["error_message"]:
                print(f"Error: {result['error_message']}")
    else:
        print("\n=== Formal Proof ===")
        print(result["formal_proof"])
        print("===================\n")
        
        # Print pattern and domain information
        print(f"Pattern: {result['pattern']}")
        print(f"Domain: {result['domain']}")
        
        # Print verification result
        print(f"Verification: {'Success' if result['verified'] else 'Failed'}")
        if not result["verified"] and result["error_message"]:
            print(f"Error: {result['error_message']}")

if __name__ == "__main__":
    sys.exit(main())