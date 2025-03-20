"""
Web application interface for the proof translation system.
Provides a web-based UI for translating informal proofs to formal proofs.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from nlp import parse_mathematical_proof
from nlp.domain_detector import DomainDetector
from nlp.pattern_recognizer import recognize_pattern
from translation.hybrid_translator import HybridTranslator
from backends.backend_interface import BackendRegistry
from utils.error_handler import handle_error

logger = logging.getLogger("formal_verification")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Define API models
class ProofInput(BaseModel):
    """Model for proof input from the API."""
    proof_text: str
    target_prover: str = "coq"
    use_llm: bool = False

class TranslationResponse(BaseModel):
    """Model for proof translation response."""
    formal_proof: str
    verification_success: bool
    domain_info: Dict[str, Any]
    pattern_info: Dict[str, Any]
    error_message: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="Informal Verfication",
    description="Translate informal mathematical proofs to formal proofs in Coq or Lean",
    version="1.0.0"
)

# Set up templates directory for HTML rendering
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# Set up static files directory for CSS, JS, etc.
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create HTML template if it doesn't exist
index_template_path = os.path.join(templates_dir, 'index.html')
if not os.path.exists(index_template_path):
    with open(index_template_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Informal Verification</title>
    <link rel="stylesheet" href="{{ url_for('static', path='/styles.css') }}">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="container">
        <h1>Informal Verification</h1>
        <p>Translate informal mathematical proofs to formal proofs in Coq or Lean.</p>
        
        <form id="proofForm" method="post" action="/translate">
            <div class="form-group">
                <label for="proofText">Proof Text:</label>
                <textarea id="proofText" name="proof_text" rows="10" placeholder="Enter your informal proof here...">{{ proof_text }}</textarea>
            </div>
            
            <div class="form-group">
                <label for="targetProver">Target Theorem Prover:</label>
                <select id="targetProver" name="target_prover">
                    <option value="coq" {% if target_prover == "coq" %}selected{% endif %}>Coq</option>
                    <option value="lean" {% if target_prover == "lean" %}selected{% endif %}>Lean</option>
                </select>
            </div>
            
        <div class="form-group">
            <label for="useLLM">
                <input type="checkbox" id="useLLM" name="use_llm" value="true" {% if use_llm %}checked{% endif %}>
                Use Language Model Assistance
            </label>
        </div>
            
            <div class="button-group">
                <button type="submit">Translate</button>
                <button type="button" id="clearButton">Clear</button>
            </div>
        </form>
        
        {% if formal_proof %}
        <div class="result-section">
            <h2>Translation Result</h2>
            
            <div class="info-panel">
                <div class="info-box">
                    <h3>Domain</h3>
                    <p>{{ domain_info.primary_domain }}</p>
                </div>
                
                <div class="info-box">
                    <h3>Proof Pattern</h3>
                    <p>{{ pattern_info.primary_pattern.name }}</p>
                </div>
                
                <div class="info-box verification-{{ "success" if verification_success else "failure" }}">
                    <h3>Verification</h3>
                    <p>{{ "Success ✓" if verification_success else "Failed ✗" }}</p>
                </div>
            </div>
            
            <div class="code-block">
                <div class="code-header">
                    <h3>{{ target_prover|upper }} Proof</h3>
                    <button id="copyButton">Copy</button>
                </div>
                <pre id="formalProof"><code>{{ formal_proof }}</code></pre>
            </div>
            
            {% if error_message %}
            <div class="error-message">
                <h3>Error Message</h3>
                <pre>{{ error_message }}</pre>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
    
    <script src="{{ url_for('static', path='/script.js') }}"></script>
</body>
</html>
""")

# Create CSS file if it doesn't exist
css_path = os.path.join(static_dir, 'styles.css')
if not os.path.exists(css_path):
    with open(css_path, 'w') as f:
        f.write("""/* General styles */
/* Reset and base styles */
/* Reset and base styles */
/* Reset and base styles */
/* Reset and base styles */
/* Reset and base styles */
/* Reset and base styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}

body {
  background: linear-gradient(to bottom, #f5f7fa, #ffffff);
  color: #333;
  line-height: 1.6;
}

/* Main container with a subtle luminous overlay */
.container {
  max-width: 1200px;
  margin: 40px auto;
  padding: 40px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
}

.container::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle, rgba(25,118,210,0.05) 0%, transparent 70%);
  pointer-events: none;
  animation: subtleGlow 20s linear infinite;
  opacity: 0.5;
}

@keyframes subtleGlow {
  0% { transform: scale(1) rotate(0deg); }
  50% { transform: scale(1.05) rotate(5deg); }
  100% { transform: scale(1) rotate(0deg); }
}

/* Headings with a delicate glow */
h1, h2, h3 {
  margin-bottom: 20px;
  color: #222;
  font-weight: 500;
  text-shadow: 0 0 3px rgba(25,118,210,0.2);
}

/* Form styles */
.form-group {
  margin-bottom: 25px;
}

label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #555;
}

textarea, select {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 6px;
  font-size: 16px;
  background: #fff;
  color: #333;
  transition: border-color 0.3s, box-shadow 0.3s;
}

textarea:focus, select:focus {
  border-color: #1976d2;
  box-shadow: 0 0 5px rgba(25,118,210,0.3);
}

textarea {
  resize: vertical;
}

.button-group {
  display: flex;
  gap: 15px;
}

/* Buttons with modern gradient and subtle glow */
button {
  padding: 12px 25px;
  background: #1976d2;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: background 0.3s, transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

button:hover {
  background: #1565c0;
  transform: translateY(-2px);
  box-shadow: 0 0 8px rgba(25,118,210,0.4);
}

#clearButton {
  background: #d32f2f;
}

#clearButton:hover {
  background: #c62828;
}

/* Result section */
.result-section {
  margin-top: 40px;
  background: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.info-panel {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 25px;
}

.info-box {
  flex: 1;
  min-width: 220px;
  padding: 20px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #ddd;
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
}

.info-box:hover {
  transform: translateY(-3px);
  box-shadow: 0 0 6px rgba(25,118,210,0.2);
}

.verification-success {
  background: #e8f5e9;
  border: 1px solid #a5d6a7;
  color: #2e7d32;
}

.verification-failure {
  background: #ffebee;
  border: 1px solid #ef9a9a;
  color: #c62828;
}

/* Code block */
.code-block {
  background: #f5f5f5;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #e0e0e0;
}

.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: #eeeeee;
  border-bottom: 1px solid #e0e0e0;
}

.code-header h3 {
  margin: 0;
  color: #333;
  text-shadow: 0 0 2px rgba(25,118,210,0.2);
}

.code-header button {
  padding: 6px 12px;
  font-size: 14px;
  background: #1976d2;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.code-header button:hover {
  background: #1565c0;
}

pre {
  padding: 15px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  line-height: 1.5;
  background: #f5f5f5;
  color: #333;
}

.error-message {
  margin-top: 25px;
  padding: 20px;
  background: #fff1f1;
  border-left: 4px solid #d32f2f;
  border-radius: 4px;
  color: #c62828;
}

/* Responsive styles */
@media (max-width: 768px) {
  .info-panel {
    flex-direction: column;
  }
  
  .info-box {
    min-width: 100%;
  }
}

""")

# Create JavaScript file if it doesn't exist
js_path = os.path.join(static_dir, 'script.js')
if not os.path.exists(js_path):
    with open(js_path, 'w') as f:
        f.write("""document.addEventListener('DOMContentLoaded', function() {
    // Copy button functionality
    const copyButton = document.getElementById('copyButton');
    if (copyButton) {
        copyButton.addEventListener('click', function() {
            const formalProof = document.getElementById('formalProof');
            const range = document.createRange();
            range.selectNode(formalProof);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            
            try {
                const successful = document.execCommand('copy');
                const msg = successful ? 'Copied!' : 'Failed to copy!';
                copyButton.textContent = msg;
                
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                }, 2000);
            } catch (err) {
                console.error('Unable to copy', err);
                copyButton.textContent = 'Error!';
                
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                }, 2000);
            }
            
            window.getSelection().removeAllRanges();
        });
    }
    
    // Clear button functionality
    const clearButton = document.getElementById('clearButton');
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            document.getElementById('proofText').value = '';
        });
    }
    
    // Form validation
    const proofForm = document.getElementById('proofForm');
    if (proofForm) {
        proofForm.addEventListener('submit', function(event) {
            const proofText = document.getElementById('proofText').value.trim();
            
            if (!proofText) {
                event.preventDefault();
                alert('Please enter a proof before submitting.');
            }
        });
    }
});
""")

# Background task to process and translate proofs using the hybrid translator
async def process_proof(
    proof_text: str,
    target_prover: str,
    use_llm: bool
) -> Dict[str, Any]:
    """
    Process and translate a proof as a background task using the hybrid translator.
    
    Args:
        proof_text: The proof text
        target_prover: The target theorem prover
        use_llm: Whether to use language model assistance
        
    Returns:
        Dictionary with translation results
    """
    try:
        # Parse the proof
        parsed_info = parse_mathematical_proof(proof_text)
        
        # Extract theorem and proof
        theorem_text = parsed_info["theorem_text"]
        proof_text = parsed_info["proof_text"]
        
        # If no separate theorem and proof were extracted, use the entire text as both
        if not theorem_text and not proof_text:
            theorem_text = proof_text = parsed_info["original_text"]
        
        # Create the hybrid translator
        translator = HybridTranslator(target_prover=target_prover, use_llm=use_llm)
        
        # Detect domain and pattern for the UI
        domain_detector = DomainDetector()
        domain_info = domain_detector.detect_domain(theorem_text, proof_text)
        pattern_info = recognize_pattern(proof_text)
        
        # Translate using the hybrid translator
        logger.info(f"Translating using the hybrid translator (use_llm: {use_llm})")
        result = translator.translate(theorem_text, proof_text)
        
        # Get the verification result
        formal_proof = result["formal_proof"]
        verified = result.get("verified", False)
        error_message = result.get("error_message")
        
        # Use domain and pattern from result if available, otherwise from detection
        domain = result.get("domain", domain_info.get("primary_domain", ""))
        pattern = result.get("pattern", pattern_info.get("primary_pattern", {}).get("name", ""))
        
        # Prepare response
        return {
            "formal_proof": formal_proof,
            "verification_success": verified,
            "domain_info": {"primary_domain": domain},
            "pattern_info": {"primary_pattern": {"name": pattern}},
            "error_message": error_message
        }
    
    except Exception as e:
        handle_error(e, "translation")
        logger.error(f"Error in process_proof: {str(e)}")
        return {
            "formal_proof": f"# Error: {str(e)}",
            "verification_success": False,
            "domain_info": {},
            "pattern_info": {"primary_pattern": {"name": "unknown"}},
            "error_message": str(e)
        }

# API routes
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """
    Render the home page.
    
    Args:
        request: The request object
        
    Returns:
        HTML response
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "proof_text": "",
            "target_prover": "coq",
            "use_llm": False
        }
    )

@app.post("/translate", response_class=HTMLResponse)
async def translate_form(
    request: Request,
    background_tasks: BackgroundTasks,
    proof_text: str = Form(...),
    target_prover: str = Form("coq"),
    use_llm: bool = Form(False)
):
    """
    Handle form submission for translation.
    
    Args:
        request: The request object
        background_tasks: Background tasks
        proof_text: The proof text
        target_prover: The target theorem prover
        use_llm: Whether to use language model assistance
        
    Returns:
        HTML response
    """
    # Log the received parameters
    logger.info(f"Translation request received: target_prover={target_prover}, use_llm={use_llm}")
    
    # Check LLM configuration if needed
    if use_llm:
        try:
            from llm.openai_client import verify_openai_setup
            is_configured, message = verify_openai_setup()
            logger.info(f"LLM configuration: {message}")
            
            if not is_configured:
                logger.warning("LLM assistance requested but not properly configured")
        except ImportError:
            logger.warning("LLM modules not available, proceeding without LLM assistance")
            use_llm = False
    
    # Process the proof
    result = await process_proof(proof_text, target_prover, use_llm)
    
    # Render the template with results
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "proof_text": proof_text,
            "target_prover": target_prover,
            "use_llm": use_llm,
            "formal_proof": result["formal_proof"],
            "verification_success": result["verification_success"],
            "domain_info": result["domain_info"],
            "pattern_info": result["pattern_info"],
            "error_message": result["error_message"]
        }
    )

@app.post("/api/translate")
async def translate_api(input_data: ProofInput):
    """
    API endpoint for proof translation.
    
    Args:
        input_data: The input data
        
    Returns:
        JSON response with translation results
    """
    try:
        # Process the proof
        result = await process_proof(
            input_data.proof_text,
            input_data.target_prover,
            input_data.use_llm
        )
        
        # Return the translation response
        return TranslationResponse(
            formal_proof=result["formal_proof"],
            verification_success=result["verification_success"],
            domain_info=result["domain_info"],
            pattern_info=result["pattern_info"],
            error_message=result["error_message"]
        )
    
    except Exception as e:
        handle_error(e, "translation")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/domain")
async def detect_domain_api(input_data: ProofInput):
    """
    API endpoint for domain detection.
    
    Args:
        input_data: The input data
        
    Returns:
        JSON response with domain information
    """
    try:
        # Parse the proof
        parsed_info = parse_mathematical_proof(input_data.proof_text)
        
        # Extract theorem and proof
        theorem_text = parsed_info["theorem_text"]
        proof_text = parsed_info["proof_text"]
        
        # If no separate theorem and proof were extracted, use the entire text as both
        if not theorem_text and not proof_text:
            theorem_text = proof_text = parsed_info["original_text"]
        
        # Detect domain
        domain_info = detect_domain(theorem_text, proof_text)
        
        return domain_info
    
    except Exception as e:
        handle_error(e, "domain_detection")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pattern")
async def recognize_pattern_api(input_data: ProofInput):
    """
    API endpoint for pattern recognition.
    
    Args:
        input_data: The input data
        
    Returns:
        JSON response with pattern information
    """
    try:
        # Parse the proof
        parsed_info = parse_mathematical_proof(input_data.proof_text)
        
        # Extract proof
        proof_text = parsed_info["proof_text"]
        
        # If no separate proof was extracted, use the entire text
        if not proof_text:
            proof_text = parsed_info["original_text"]
        
        # Recognize pattern
        pattern_info = recognize_pattern(proof_text)
        
        return pattern_info
    
    except Exception as e:
        handle_error(e, "pattern_recognition")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backend_info")
async def get_backend_info():
    """
    API endpoint for backend information.
    
    Returns:
        JSON response with backend information
    """
    backends = BackendRegistry.list_backends()
    backend_info = {}
    
    for backend_name in backends:
        try:
            backend = BackendRegistry.get_backend(backend_name)
            backend_info[backend_name] = {
                "name": backend.name,
                "version": backend.version,
                "installed": backend.is_installed(),
                "supported_libraries": backend.supported_libraries
            }
        except Exception as e:
            backend_info[backend_name] = {
                "name": backend_name,
                "error": str(e),
                "installed": False
            }
    
    return {"backends": backend_info}

# Run the app if executed directly
if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the web application for proof translation.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Run the app
    uvicorn.run("ui.web_app:app", host=args.host, port=args.port, reload=args.reload)