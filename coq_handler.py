import subprocess
import os
import re
import tempfile
from openai import OpenAI

def clean_llm_output(llm_output):
    """
    Remove any markdown code block markers or extraneous whitespace.
    """
    cleaned = llm_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove the first and last lines if they are markdown markers
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def run_coqc(coq_proof, filename="proof.v"):
    """
    Save the Coq proof to a file and run the Coq compiler (coqc).
    Returns a tuple: (success, stdout, error_msg).
    """
    with open(filename, "w") as f:
        f.write(coq_proof)
    print(f"Saved Coq proof as {filename}")
    try:
        result = subprocess.run(
            ["coqc", filename],
            capture_output=True,
            text=True,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        return False, None, "Coq verification timed out."
    except FileNotFoundError:
        return False, None, "Coq compiler (coqc) not found. Please ensure Coq is installed."
    
    if result.returncode == 0:
        return True, result.stdout, None
    else:
        return False, None, result.stderr

def parse_error_category(error_msg):
    """
    Categorize the Coq error message based on known patterns.
    """
    if "Syntax error" in error_msg:
        return "syntax"
    elif "was not found" in error_msg:
        return "missing_reference"
    elif "No such contradiction" in error_msg:
        return "tactic_error"
    elif "Unable to unify" in error_msg:
        return "unification"
    elif "No matching clauses for match" in error_msg:
        return "pattern_matching"
    elif "Cannot solve this goal" in error_msg:
        return "incomplete_proof"
    else:
        return "general"

def severity_for_category(category):
    """
    Map error category to a severity level.
    """
    if category in ["syntax", "missing_reference"]:
        return "high"
    elif category in ["tactic_error", "unification", "pattern_matching", "incomplete_proof"]:
        return "medium"
    else:
        return "low"

def fix_with_llm(coq_proof, error_msg, category, model="gpt-4-turbo"):
    """
    Use the LLM to attempt to fix a given error.
    The prompt includes the current proof, the error message, its category, and severity.
    """
    severity = severity_for_category(category)
    prompt = f"""
The following Coq proof encountered an error.

Coq Proof:
---
{coq_proof}
---

Error (severity: {severity}, category: {category}):
---
{error_msg}
---

Your task:
- Identify the issue indicated by the error message.
- Correct ONLY the problematic part of the proof without altering correct parts.
- Output ONLY the corrected complete Coq proof script (with no additional commentary).

Proceed now:
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Coq assistant that fixes Coq proofs based on error messages."},
            {"role": "user", "content": prompt}
        ]
    )
    corrected_code = response.choices[0].message.content
    return clean_llm_output(corrected_code)

def iterative_proof_fix(coq_proof, max_iterations=5, filename="proof.v"):
    """
    Main iterative feedback loop:
    - Run Coq on the current proof.
    - If errors occur, categorize and prompt the LLM for a fix.
    - Loop until the proof is verified or the max iteration limit is reached.
    """
    iteration = 0
    current_proof = coq_proof
    last_error = None

    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}: Verifying proof...")
        success, output, error_msg = run_coqc(current_proof, filename)
        if success:
            print("✅ Proof verified successfully!")
            return current_proof
        else:
            print("❌ Coq error encountered:")
            print(error_msg)
            # Prevent infinite loops if the error hasn't changed
            if last_error and last_error.strip() == error_msg.strip():
                print("Error unchanged from previous iteration. Stopping further attempts.")
                break
            last_error = error_msg

            # Categorize the error and prompt the LLM for a fix.
            category = parse_error_category(error_msg)
            print(f"Detected error category: {category}")
            print("Attempting automatic LLM fix...")
            fixed_proof = fix_with_llm(current_proof, error_msg, category)
            if fixed_proof.strip() == current_proof.strip():
                print("LLM did not propose any changes. Stopping iteration.")
                break
            current_proof = fixed_proof
            print("LLM fix applied. Re-verifying proof...")

    print("Failed to verify proof within the iteration limit.")
    return current_proof

def save_and_verify_coq(coq_proof, filename="proof.v"):
    """
    Legacy function to save a proof and verify it once.
    For full automation, use iterative_proof_fix instead.
    """
    with open(filename, "w") as f:
        f.write(coq_proof)
    
    print(f"Saved Coq proof as {filename}")
    
    has_admits = "admit" in coq_proof or "Admitted" in coq_proof
    try:
        result = subprocess.run(
            ["coqc", filename],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            if has_admits:
                print("✅ Coq accepted the proof structure, but it contains admits/placeholders that need to be filled in.")
            else:
                print("✅ Coq successfully verified the complete proof!")
            return True
        else:
            print("❌ Coq found issues with the proof:")
            print(result.stderr)
            analyze_coq_error(result.stderr, coq_proof)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Coq verification timed out")
        return False
    except FileNotFoundError:
        print("❌ Coq compiler (coqc) not found. Please ensure Coq is installed.")
        return False

def analyze_coq_error(error_msg, coq_proof, filename="proof.v"):
    """
    Legacy error analysis function that prints details about Coq errors.
    With iterative_proof_fix, this is replaced by the automated feedback loop.
    """
    if "Syntax error" in error_msg:
        print("→ Syntax Error: Check for typos or incorrect Coq syntax")
        line_match = re.search(r"line (\d+)", error_msg)
        if line_match:
            line_num = int(line_match.group(1))
            lines = coq_proof.split('\n')
            if line_num <= len(lines):
                print(f"  Problematic line: {lines[line_num - 1]}")
    elif "The reference" in error_msg and "was not found" in error_msg:
        ref_match = re.search(r"The reference ([^ ]+) was not found", error_msg)
        if ref_match:
            missing_ref = ref_match.group(1)
            print(f"→ Missing Reference: '{missing_ref}' not found.")
            print("Attempting automatic LLM fix for missing import...")
            corrected_coq_code = fix_with_llm(coq_proof, error_msg, "missing_reference")
            with open(filename, "w") as f:
                f.write(corrected_coq_code)
            result = subprocess.run(
                ["coqc", filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✅ LLM successfully fixed the proof!")
            else:
                print("❌ LLM attempted fix, but Coq still reports an error:")
                print(result.stderr)
    elif "Unable to unify" in error_msg:
        print("→ Type Mismatch: Coq expected one type but found another")
    elif "No matching clauses for match" in error_msg:
        print("→ Incomplete Pattern Matching: Not all cases are covered in a match expression")
    elif "Cannot solve this goal" in error_msg:
        print("→ Incomplete Proof: A tactic failed to solve a goal")
        print("  Try using more specific tactics or breaking the proof into smaller steps.")
    else:
        print("→ General Error: Could not determine the specific issue")
        print("  Try simplifying the proof or breaking it down into smaller steps.")
    
    print("\nSuggestions:")
    print("1. Use 'Admitted.' temporarily to skip problematic parts")
    print("2. Check Coq documentation for correct syntax and tactics")
    print("3. Try a simpler version of the proof first")

def interactive_coq_session(coq_proof):
    """
    Create a temporary file for interactive Coq sessions.
    """
    with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(coq_proof.encode())
    
    print(f"\nInteractive Session:")
    print(f"1. A temporary file has been created at: {temp_filename}")
    print(f"2. You can run an interactive Coq session with:")
    print(f"   coqtop -l {temp_filename}")
    print(f"3. Inside coqtop, you can step through the proof with commands like:")
    print(f"   - 'Next.' to advance to the next step")
    print(f"   - 'Show.' to see the current goal")
    print(f"   - 'Undo.' to go back one step")
    
    return temp_filename

# Example usage of the new iterative feedback loop:
if __name__ == "__main__":
    initial_proof = """
Theorem empty_set_lie_algebra: forall (A: Type) (plus mult: A -> A -> A) (inv: A -> A) (zero: A), 
  (forall x y z: A, plus x (plus y z) = plus (plus x y) z) ->
  (forall x: A, plus x zero = x) ->
  (forall x: A, plus x (inv x) = zero) ->
  (forall x y: A, plus x y = plus y x) ->
  (forall x y z: A, mult x (plus y z) = plus (mult x y) (mult x z)) ->
  (forall x: A, mult zero x = zero) ->
  (forall x: A, mult x zero = zero) ->
  (forall x y z: A, mult x (mult y z) = mult (mult x y) z) ->
  (forall x: A, mult x (inv x) = zero) ->
  (forall x: A, mult (inv x) x = zero) ->
  (forall x: A, False).
Proof.
  intros A plus mult inv zero H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 x.
  contradiction.
Qed.
"""
    final_proof = iterative_proof_fix(initial_proof, max_iterations=5)
    print("\nFinal proof script:")
    print(final_proof)
