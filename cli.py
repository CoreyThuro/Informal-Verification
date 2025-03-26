"""
Updated CLI module with web UI support.
"""

import argparse
import sys
import os
import uvicorn

from translator import ProofTranslator

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate informal mathematical proofs to Coq."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Translation command
    translate_parser = subparsers.add_parser("translate", help="Translate a proof")
    
    input_group = translate_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--theorem", "-t",
        help="The theorem statement"
    )
    input_group.add_argument(
        "--file", "-f",
        help="File containing the theorem and proof (separate with 'Proof:')"
    )
    
    translate_parser.add_argument(
        "--proof", "-p",
        help="The proof text (required if --theorem is used)"
    )
    
    translate_parser.add_argument(
        "--output", "-o",
        help="Output file for the formal proof (default: stdout)"
    )
    
    translate_parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the generated proof"
    )
    
    # Web server command
    web_parser = subparsers.add_parser("web", help="Start the web UI")
    web_parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to run the server on (default: 127.0.0.1)"
    )
    web_parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if args.command == "translate":
        return translate_command(args)
    elif args.command == "web":
        return web_command(args)
    else:
        parser.print_help()
        return 1

def translate_command(args):
    """Handle translate command."""
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
    
    return 0

def web_command(args):
    """Handle web command."""
    print(f"Starting web server at http://{args.host}:{args.port}")
    
    # Import the web app and run it with uvicorn
    import importlib.util
    import sys
    
    # Check if web_app.py exists, if not, create it
    web_app_path = os.path.join(os.path.dirname(__file__), "web_app.py")
    if not os.path.exists(web_app_path):
        print("Error: web_app.py not found. Please make sure it exists in the project directory.")
        return 1
    
    # Run the server
    uvicorn.run("web_app:app", host=args.host, port=args.port)
    return 0

if __name__ == "__main__":
    sys.exit(main())