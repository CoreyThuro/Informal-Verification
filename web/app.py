"""
FastAPI web interface for the proof translator.
"""

import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from translator import ProofTranslator

# Initialize FastAPI app
app = FastAPI(title="Proof Translator")

# Set up template and static directories
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Initialize translator
translator = ProofTranslator()

class TranslationRequest(BaseModel):
    theorem: str
    proof: str

class TranslationResponse(BaseModel):
    formal_proof: str
    verified: bool
    error_message: Optional[str] = None
    pattern: str
    domain: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate/", response_model=TranslationResponse)
async def translate_proof(translation_request: TranslationRequest):
    """
    Translate an informal proof to Coq.
    """
    result = translator.translate(
        translation_request.theorem, 
        translation_request.proof
    )
    
    return result

@app.post("/translate-form", response_class=HTMLResponse)
async def translate_form(
    request: Request,
    theorem: str = Form(...),
    proof: str = Form(...)
):
    """
    Handle form submission and display results.
    """
    result = translator.translate(theorem, proof)
    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "result": result,
            "theorem": theorem,
            "proof": proof
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)