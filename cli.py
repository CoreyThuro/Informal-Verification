"""
Enhanced command-line interface for proof translation.
"""

import argparse
import sys
import os
import time
import json

from translator import ProofTranslator

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate informal mathematical proofs to Coq."
    )
    
    # Main subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Translate command
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
    translate_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple proofs")
    batch_parser.add_argument(
        "directory",
        help="Directory containing proof files"
    )
    batch_parser.add_argument(
        "--output", "-o",
        help="Output directory for the formal proofs"
    )
    batch_parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the generated proofs"
    )
    batch_parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Generate a summary report"
    )
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Start the web interface")
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    web_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle translate command
    if args.command == "translate":
        return handle_translate(args)
    
    # Handle batch command
    elif args.command == "batch":
        return handle_batch(args)
    
    # Handle web command
    elif args.command == "web":
        return handle_web(args)

def handle_translate(args):
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
    
    # Start translator
    print("Translating proof...", file=sys.stderr)
    translator = ProofTranslator()
    
    start_time = time.time()
    result = translator.translate(theorem_text, proof_text)
    end_time = time.time()
    
    # Format result
    if args.json:
        # JSON output
        json_result = {
            "formal_proof": result["formal_proof"],
            "verified": result["verified"],
            "pattern": result["pattern"],
            "domain": result["domain"],
            "error_message": result["error_message"],
            "translation_time": round(end_time - start_time, 2)
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(json_result, f, indent=2)
            print(f"Result written to {args.output}")
        else:
            print(json.dumps(json_result, indent=2))
    else:
        # Text output
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
            print(f"Translation time: {round(end_time - start_time, 2)} seconds")
            
            # Print verification result
            print(f"Verification: {'Success' if result['verified'] else 'Failed'}")
            if not result["verified"] and result["error_message"]:
                print(f"Error: {result['error_message']}")
    
    return 0

def handle_batch(args):
    """Handle batch command."""
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return 1
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Find proof files
    proof_files = [f for f in os.listdir(args.directory) if f.endswith('.txt')]
    if not proof_files:
        print(f"No proof files found in {args.directory}")
        return 1
    
    print(f"Found {len(proof_files)} proof files. Processing...")
    
    # Initialize translator
    translator = ProofTranslator()
    
    # Process each file
    results = []
    for filename in proof_files:
        filepath = os.path.join(args.directory, filename)
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Split into theorem and proof
            parts = content.split("Proof:", 1)
            if len(parts) < 2:
                print(f"Error in {filename}: File must contain 'Proof:' separator")
                continue
                
            theorem_text = parts[0].strip()
            proof_text = parts[1].strip()
            
            # Translate
            print(f"Translating {filename}...", file=sys.stderr)
            start_time = time.time()
            result = translator.translate(theorem_text, proof_text)
            end_time = time.time()
            
            # Write output
            if args.output:
                output_filename = os.path.splitext(filename)[0] + ".v"
                output_path = os.path.join(args.output, output_filename)
                with open(output_path, "w") as f:
                    f.write(result["formal_proof"])
            
            # Store result for summary
            results.append({
                "filename": filename,
                "pattern": result["pattern"],
                "domain": result["domain"],
                "verified": result["verified"],
                "translation_time": round(end_time - start_time, 2)
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Generate summary if requested
    if args.summary:
        summary = {
            "total_files": len(proof_files),
            "processed_files": len(results),
            "verified_proofs": sum(1 for r in results if r["verified"]),
            "pattern_counts": {},
            "domain_counts": {},
            "average_time": sum(r["translation_time"] for r in results) / len(results) if results else 0
        }
        
        # Count patterns and domains
        for result in results:
            pattern = result["pattern"]
            domain = result["domain"]
            
            if pattern in summary["pattern_counts"]:
                summary["pattern_counts"][pattern] += 1
            else:
                summary["pattern_counts"][pattern] = 1
                
            if domain in summary["domain_counts"]:
                summary["domain_counts"][domain] += 1
            else:
                summary["domain_counts"][domain] = 1
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total files: {summary['total_files']}")
        print(f"Processed files: {summary['processed_files']}")
        print(f"Verified proofs: {summary['verified_proofs']}")
        print(f"Average translation time: {round(summary['average_time'], 2)} seconds")
        
        print("\nPattern distribution:")
        for pattern, count in summary["pattern_counts"].items():
            print(f"  {pattern}: {count} ({round(count/len(results)*100)}%)")
        
        print("\nDomain distribution:")
        for domain, count in summary["domain_counts"].items():
            print(f"  {domain}: {count} ({round(count/len(results)*100)}%)")
        
        # Write summary to file if output directory specified
        if args.output:
            summary_path = os.path.join(args.output, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary written to {summary_path}")
    
    return 0

def handle_web(args):
    """Handle web command."""
    try:
        # Import web app here to avoid dependency for other commands
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from web.app import app
        import uvicorn
        
        print(f"Starting web server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ImportError:
        print("Error: Web dependencies not installed. Install with: pip install fastapi uvicorn jinja2")
        return 1
    except Exception as e:
        print(f"Error starting web server: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())