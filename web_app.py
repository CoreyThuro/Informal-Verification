"""
Interactive web interface for the proof translation system using FastAPI.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from translator import ProofTranslator
from patterns.recognizer import recognize_pattern
from knowledge.kb import KnowledgeBase
from coq.verifier import verify_coq_proof
from coq.feedback import apply_feedback

# Models for request data
class ProofRequest(BaseModel):
    theorem: str
    proof: str

class VerifyRequest(BaseModel):
    proof: str

class FeedbackRequest(BaseModel):
    proof: str
    error: str

# Create FastAPI app
app = FastAPI(title="Proof Translator")
translator = ProofTranslator()
kb = KnowledgeBase()

# Ensure directories exist
os.makedirs('/web/static', exist_ok=True)
os.makedirs('/web/static/js', exist_ok=True)
os.makedirs('/web/static/css', exist_ok=True)
os.makedirs('/web/templates', exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate")
async def translate(req: ProofRequest):
    """Translate a proof and return the result."""
    try:
        # Get the full translation
        result = translator.translate(req.theorem, req.proof)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(req: ProofRequest):
    """Analyze a proof without full translation."""
    try:
        # Recognize pattern and domain
        pattern, pattern_info = recognize_pattern(req.theorem, req.proof)
        domain = translator._detect_domain(req.theorem, req.proof)
        
        # Get domain and pattern info
        domain_info = kb.get_domain_info(domain)
        pattern_details = kb.get_pattern_info(pattern)
        
        # Get recommended tactics
        pattern_tactics = kb.get_pattern_tactics(pattern)
        domain_tactics = kb.get_domain_tactics(domain)
        
        # Get imports
        imports = kb.get_imports_for_domain(domain)
        
        return {
            'pattern': pattern,
            'pattern_info': pattern_info,
            'pattern_details': pattern_details,
            'domain': domain,
            'domain_info': domain_info,
            'imports': imports,
            'pattern_tactics': pattern_tactics,
            'domain_tactics': domain_tactics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step_translate")
async def step_translate(req: ProofRequest):
    """Perform a step-by-step translation."""
    try:
        # First, analyze the proof
        pattern, pattern_info = recognize_pattern(req.theorem, req.proof)
        domain = translator._detect_domain(req.theorem, req.proof)
        
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
        
        return {
            'steps': steps,
            'final_proof': full_proof
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify(req: VerifyRequest):
    """Verify a Coq proof."""
    try:
        verified, error = verify_coq_proof(req.proof)
        
        return {
            'verified': verified,
            'error': error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply_feedback")
async def apply_feedback_endpoint(req: FeedbackRequest):
    """Apply feedback to fix errors in a proof."""
    try:
        fixed_proof = apply_feedback(req.proof, req.error)
        
        return {
            'fixed_proof': fixed_proof
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))