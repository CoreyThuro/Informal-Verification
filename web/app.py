"""
Interactive web interface for the proof translation system.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import json
from translator import ProofTranslator
from patterns.recognizer import recognize_pattern
from knowledge.kb import KnowledgeBase

app = Flask(__name__)
translator = ProofTranslator()
kb = KnowledgeBase()

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Translate a proof and return the result."""
    data = request.get_json()
    theorem_text = data.get('theorem', '')
    proof_text = data.get('proof', '')
    
    # Get the full translation
    result = translator.translate(theorem_text, proof_text)
    
    return jsonify(result)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a proof without full translation."""
    data = request.get_json()
    theorem_text = data.get('theorem', '')
    proof_text = data.get('proof', '')
    
    # Recognize pattern and domain
    pattern, pattern_info = recognize_pattern(theorem_text, proof_text)
    domain = translator._detect_domain(theorem_text, proof_text)
    
    # Get domain and pattern info
    domain_info = kb.get_domain_info(domain)
    pattern_details = kb.get_pattern_info(pattern)
    
    # Get recommended tactics
    pattern_tactics = kb.get_pattern_tactics(pattern)
    domain_tactics = kb.get_domain_tactics(domain)
    
    # Get imports
    imports = kb.get_imports_for_domain(domain)
    
    return jsonify({
        'pattern': pattern,
        'pattern_info': pattern_info,
        'pattern_details': pattern_details,
        'domain': domain,
        'domain_info': domain_info,
        'imports': imports,
        'pattern_tactics': pattern_tactics,
        'domain_tactics': domain_tactics
    })

@app.route('/step_translate', methods=['POST'])
def step_translate():
    """Perform a step-by-step translation."""
    data = request.get_json()
    theorem_text = data.get('theorem', '')
    proof_text = data.get('proof', '')
    
    # First, analyze the proof
    pattern, pattern_info = recognize_pattern(theorem_text, proof_text)
    domain = translator._detect_domain(theorem_text, proof_text)
    
    # Get imports
    imports = kb.get_imports_for_domain(domain)
    
    # Generate steps for translation
    steps = []
    
    # Step 1: Pattern and domain detection
    steps.append({
        'name': 'Detect pattern and domain',
        'description': f'Detected pattern: {pattern}, domain: {domain}',
        'code': f'# Pattern: {pattern}\n# Domain: {domain}'
    })
    
    # Step 2: Add imports
    imports_code = '\n'.join(imports)
    steps.append({
        'name': 'Add necessary imports',
        'description': f'Adding imports for domain {domain}',
        'code': imports_code
    })
    
    # Step 3: Generate theorem statement
    variables = pattern_info.get('variables', ['n'])
    vars_str = ", ".join([f"{v}: nat" for v in variables])
    theorem_code = f"Theorem example: forall {vars_str}, P({', '.join(variables)})."
    steps.append({
        'name': 'Generate theorem statement',
        'description': 'Translating the informal theorem to Coq syntax',
        'code': theorem_code
    })
    
    # Step 4: Generate proof structure
    proof_structure = "Proof.\n  intros.\n\n  (* Proof body will go here *)\n\nQed."
    steps.append({
        'name': 'Generate proof structure',
        'description': 'Creating basic proof structure',
        'code': proof_structure
    })
    
    # Step 5: Add tactics based on pattern
    tactics = []
    if pattern == "evenness":
        variable = pattern_info.get('variable', variables[0] if variables else 'n')
        tactics.append(f"  exists {variable}.")
        tactics.append("  ring.")
    elif pattern == "induction":
        variable = pattern_info.get('variable', variables[0] if variables else 'n')
        tactics.append(f"  induction {variable}.")
        tactics.append(f"  (* Base case: {variable} = 0 *)")
        tactics.append("  simpl.")
        tactics.append("  auto.")
        tactics.append(f"  (* Inductive step: {variable} = S n *)")
        tactics.append("  simpl.")
        tactics.append(f"  rewrite IH{variable}.")
        tactics.append("  auto.")
    elif pattern == "contradiction":
        tactics.append("  (* Proof by contradiction *)")
        tactics.append("  assert (H : ~P).")
        tactics.append("  {")
        tactics.append("    intro contra.")
        tactics.append("    (* Derive contradiction *)")
        tactics.append("    contradiction.")
        tactics.append("  }")
        tactics.append("  contradiction.")
    elif pattern == "cases":
        case_var = pattern_info.get('case_var', variables[0] if variables else 'n')
        tactics.append(f"  (* Case analysis on {case_var} *)")
        tactics.append(f"  destruct {case_var}.")
        tactics.append(f"  (* Case: {case_var} = 0 *)")
        tactics.append("  simpl.")
        tactics.append("  auto.")
        tactics.append(f"  (* Case: {case_var} = S n *)")
        tactics.append("  simpl.")
        tactics.append("  auto.")
    else:  # Direct proof
        tactics.append("  (* Direct proof *)")
        tactics.append("  auto.")
    
    tactics_code = '\n'.join(tactics)
    steps.append({
        'name': 'Add proof tactics',
        'description': f'Adding tactics for {pattern} pattern',
        'code': tactics_code
    })
    
    # Step 6: Complete proof
    full_proof = imports_code + "\n\n" + theorem_code + "\n" + "Proof.\n  intros.\n" + '\n'.join(tactics) + "\nQed."
    steps.append({
        'name': 'Complete proof',
        'description': 'Final Coq proof',
        'code': full_proof
    })
    
    return jsonify({
        'steps': steps,
        'final_proof': full_proof
    })

@app.route('/verify', methods=['POST'])
def verify():
    """Verify a Coq proof."""
    from coq.verifier import verify_coq_proof
    
    data = request.get_json()
    proof_script = data.get('proof', '')
    
    verified, error = verify_coq_proof(proof_script)
    
    return jsonify({
        'verified': verified,
        'error': error
    })

@app.route('/apply_feedback', methods=['POST'])
def apply_feedback():
    """Apply feedback to fix errors in a proof."""
    from coq.feedback import apply_feedback
    
    data = request.get_json()
    proof_script = data.get('proof', '')
    error_message = data.get('error', '')
    
    fixed_proof = apply_feedback(proof_script, error_message)
    
    return jsonify({
        'fixed_proof': fixed_proof
    })

if __name__ == '__main__':
    app.run(debug=True)