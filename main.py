import sys
import argparse
import os
from parser import parse_proof
from hybrid_translator import translate_to_coq
from coq_handler import save_and_verify_coq, interactive_coq_session

# Clean up Coq artifacts after running
def cleanup_coq_artifacts():
    artifacts = ["proof.glob", "proof.vo", "proof.vok", "proof.vos"]
    for artifact in artifacts:
        if os.path.exists(artifact):
            try:
                os.remove(artifact)
                print(f"Removed {artifact}")
            except Exception as e:
                print(f"Failed to remove {artifact}: {e}")
        else:
            print(f"{artifact} does not exist")

def analyze_informal_proof(proof_text):
    """Analyze an informal proof and provide statistics"""
    print("\n=== Proof Analysis ===")
    print(f"Length: {len(proof_text)} characters, {len(proof_text.split())} words")
    
    # Count sentences
    sentences = [s for s in proof_text.split('.') if s.strip()]
    print(f"Sentences: {len(sentences)}")
    
    # Count potential mathematical expressions
    import re
    math_expr_pattern = r'\b([a-zA-Z][a-zA-Z0-9]*(?:\s*[\+\-\*\/\^\=]\s*[a-zA-Z0-9]+)+)'
    math_expressions = re.findall(math_expr_pattern, proof_text)
    print(f"Mathematical expressions detected: {len(math_expressions)}")
    if math_expressions:
        print(f"  Examples: {', '.join(math_expressions[:3])}")
    
    # Identify keywords
    keywords = ["assume", "let", "suppose", "then", "therefore", "thus", "hence", 
                "by", "because", "since", "induction", "contradiction", "case"]
    found_keywords = []
    for keyword in keywords:
        if keyword in proof_text.lower():
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"Logical keywords: {', '.join(found_keywords)}")
    
    # Potential proof method detection
    methods = {
        "induction": "Inductive proof",
        "contradiction": "Proof by contradiction",
        "case": "Case analysis",
        "cases": "Case analysis",
        "suppose": "Direct proof"
    }
    
    detected_methods = []
    for method, description in methods.items():
        if method in proof_text.lower():
            detected_methods.append(description)
    
    if detected_methods:
        print(f"Potential proof method: {', '.join(detected_methods)}")
    else:
        print("Potential proof method: Direct proof (default)")
    
    print("==================\n")

def main():
    parser = argparse.ArgumentParser(description='Translate informal proofs to Coq.')
    parser.add_argument('proof', nargs='?', help='The informal proof text')
    parser.add_argument('-f', '--file', help='Read proof from the specified file')
    parser.add_argument('-i', '--interactive', action='store_true', 
                        help='Start an interactive Coq session after translation')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM fallback, use only rule-based translation')
    parser.add_argument('--llm-first', action='store_true',
                        help='Prioritize LLM translation over rule-based approach')
    parser.add_argument('--no-feedback', action='store_true',
                        help='Disable mathematical feedback analysis')
    parser.add_argument('--model', default='gpt-4',
                        help='Specify LLM model to use (default: gpt-4)')
    parser.add_argument('--local-llm', action='store_true',
                        help='Use a locally hosted LLM instead of OpenAI')
    
    args = parser.parse_args()
    
    # Get the proof text from arguments or file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                proof_text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            return
    elif args.proof:
        proof_text = args.proof
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        proof_text = sys.argv[1]
    else:
        # No proof provided, show help
        parser.print_help()
        return
    
    # Analyze the proof
    analyze_informal_proof(proof_text)
    
    # Step 1: Parse the proof
    print("Parsing proof...")
    parsed_data = parse_proof(proof_text)
    
    if args.debug:
        parsed_statements, proof_structure = parsed_data
        print("\nParsed Statements:")
        for stmt in parsed_statements:
            print(f"  {' '.join([token[0] for token in stmt])}")
        
        print("\nIdentified Structure:")
        print(f"  Variables: {proof_structure['variables']}")
        print(f"  Expressions: {[expr for expr, _ in proof_structure['expressions']]}")
        print(f"  Assumptions: {[stmt for stmt, _ in proof_structure['assumptions']]}")
        print(f"  Conclusions: {[stmt for stmt, _ in proof_structure['conclusions']]}")
        print(f"  Proof Methods: {[method for method, _, _ in proof_structure['proof_methods']]}")
    
    # Step 2: Translate parsed proof to Coq
    print("\nTranslating to Coq...")
    
    # Configure the translator based on arguments
    if args.no_llm:
        # Import the hybrid translator but disable LLM fallback
        from hybrid_translator import HybridTranslator
        translator = HybridTranslator(fallback_to_llm=False)
        coq_proof = translator.translate(parsed_data)
    else:
        # Check for LLM configuration
        use_local_llm = args.local_llm
        
        if not use_local_llm:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY environment variable not found.")
                print("Either set this variable or use --local-llm option.")
                use_local = input("Would you like to use a local LLM instead? (y/n): ").lower() == 'y'
                if use_local:
                    use_local_llm = True
                else:
                    print("Falling back to rule-based translation only.")
                    args.no_llm = True
        
        if args.no_llm:
            # Fall back to rule-based only
            from hybrid_translator import HybridTranslator
            translator = HybridTranslator(fallback_to_llm=False)
            coq_proof = translator.translate(parsed_data)
        else:
            # Use LLM-assisted translation
            from llm_translator import LLMTranslator
            llm_translator = LLMTranslator(
                model_name=args.model,
                use_local=use_local_llm
            )
            
            # Check if we should disable mathematical feedback
            if args.no_feedback:
                # Patch the get_feedback method to always return None
                original_get_feedback = llm_translator.get_feedback
                llm_translator.get_feedback = lambda *args, **kwargs: None
            
            from hybrid_translator import HybridTranslator
            translator = HybridTranslator(
                llm_translator=llm_translator,
                fallback_to_llm=True,
                llm_first=args.llm_first
            )
            coq_proof = translator.translate(parsed_data)
    
    # Step 3: Save & verify Coq proof
    print("\nVerifying with Coq...")
    verification_result = save_and_verify_coq(coq_proof)
    
    # Step 4: Interactive session if requested
    if args.interactive:
        temp_file = interactive_coq_session(coq_proof)
        print(f"\nTemporary file for interactive session: {temp_file}")
        print("Remember to manually delete this file when you're done.")

if __name__ == "__main__":
    main()
    # cleanup_coq_artifacts()