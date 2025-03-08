import re
# coq_imports.py
"""
Comprehensive reference dictionary for Coq imports and dependencies.
Maps tactics, types, functions, and constructs to their required imports.
Built using information from the Coq standard library.
"""

# Main reference dictionary: maps references to their required imports
COQ_IMPORTS = {
    # Number types and operations
    "Z": "Require Import ZArith.",
    "R": "Require Import Reals.",
    "Q": "Require Import QArith.",
    "nat": "Require Import Arith.",
    "N": "Require Import NArith.",
    "positive": "Require Import PArith.",
    "Pos": "Require Import PArith.",
    
    # Common modules by prefix
    "Nat.": "Require Import Arith.",
    "Z.": "Require Import ZArith.",
    "N.": "Require Import NArith.",
    "R.": "Require Import Reals.",
    "Q.": "Require Import QArith.",
    "List.": "Require Import Lists.List.",
    "String.": "Require Import Strings.String.",
    "Bool.": "Require Import Bool.",
    
    # Common tactics
    "ring": "Require Import Ring.",
    "field": "Require Import Field.",
    "lia": "Require Import Lia.",
    "lra": "Require Import Lra.",
    "nia": "Require Import Lia.",
    "omega": "Require Import Omega.",
    "nsatz": "Require Import Nsatz.",
    "fourier": "Require Import Fourier.",
    "psatz": "Require Import Psatz.",
    "btauto": "Require Import Btauto.",
    "auto": "",  # Built-in
    "simpl": "",  # Built-in
    "intro": "",  # Built-in
    "intros": "",  # Built-in
    "apply": "",  # Built-in
    "rewrite": "",  # Built-in
    "reflexivity": "",  # Built-in
    "unfold": "",  # Built-in
    "destruct": "",  # Built-in
    "induction": "",  # Built-in
    "clear": "",  # Built-in
    "assert": "",  # Built-in
    "exact": "",  # Built-in
    "exists": "",  # Built-in
    "split": "",  # Built-in
    "assumption": "",  # Built-in
    "pose": "",  # Built-in
    "revert": "",  # Built-in
    "specialize": "",  # Built-in
    "transitivity": "",  # Built-in
    "symmetry": "",  # Built-in
    "contradiction": "",  # Built-in
    "trivial": "",  # Built-in
    "discriminate": "",  # Built-in
    "inversion": "",  # Built-in
    "eapply": "",  # Built-in
    "eauto": "",  # Built-in
    "tauto": "",  # Built-in
    
    # Scopes
    "Z_scope": "Open Scope Z_scope.",
    "R_scope": "Open Scope R_scope.",
    "Q_scope": "Open Scope Q_scope.",
    "N_scope": "Open Scope N_scope.",
    "nat_scope": "",  # Default scope
    "type_scope": "",  # Built-in
    "list_scope": "Require Import Lists.List.",
    "string_scope": "Require Import Strings.String.",
    
    # Additional Math Libraries and Functions
    "Arith": "Require Import Arith.",
    "ZArith": "Require Import ZArith.",
    "Bool": "Require Import Bool.",
    "List": "Require Import Lists.List.",
    "String": "Require Import Strings.String.",
    "Relations": "Require Import Relations.",
    "Sets": "Require Import Sets.",
    "Logic": "Require Import Logic.",
    "Reals": "Require Import Reals.",
    "QArith": "Require Import QArith.",
    "NArith": "Require Import NArith.",
    "PArith": "Require Import PArith.",
    "Setoid": "Require Import Setoid.",
    "ClassicalFacts": "Require Import ClassicalFacts.",
    "ClassicalChoice": "Require Import ClassicalChoice.",
    "ClassicalDescription": "Require Import ClassicalDescription.",
    "ClassicalEpsilon": "Require Import ClassicalEpsilon.",
    "Wellfounded": "Require Import Wellfounded.",
    "Program": "Require Import Program.",
    "Recdef": "Require Import Recdef.",
    "Extraction": "Require Import Extraction.",
    
    # Common specialized functions and constants
    "Nat.add": "Require Import Arith.",
    "Nat.mul": "Require Import Arith.",
    "Nat.div": "Require Import Arith.",
    "Nat.modulo": "Require Import Arith.",
    "Nat.gcd": "Require Import Arith.",
    "Nat.lcm": "Require Import Arith.",
    "Nat.pow": "Require Import Arith.",
    "Nat.log2": "Require Import Arith.",
    "Nat.sqrt": "Require Import Arith.",
    "Nat.bitwise": "Require Import Arith.",
    "Z.add": "Require Import ZArith.",
    "Z.mul": "Require Import ZArith.",
    "Z.div": "Require Import ZArith.",
    "Z.modulo": "Require Import ZArith.",
    "Z.gcd": "Require Import ZArith.",
    "Z.lcm": "Require Import ZArith.",
    "Z.pow": "Require Import ZArith.",
    "Z.log2": "Require Import ZArith.",
    "Z.sqrt": "Require Import ZArith.",
    "Z.div_eucl": "Require Import ZArith.",
    "R.add": "Require Import Reals.",
    "R.mul": "Require Import Reals.",
    "R.div": "Require Import Reals.",
    "R.pow": "Require Import Reals.",
    "R.sqrt": "Require Import Reals.",
    "sin": "Require Import Reals.",
    "cos": "Require Import Reals.",
    "tan": "Require Import Reals.",
    "exp": "Require Import Reals.",
    "ln": "Require Import Reals.",
    "Rabs": "Require Import Reals.",
    "Rinv": "Require Import Reals.",
    "Rsqr": "Require Import Reals.",
    "PI": "Require Import Reals.",
    "INR": "Require Import Reals.",
    "IZR": "Require Import Reals.",
    
    # Micromega tactics
    "zify": "Require Import Zify.",
    
    # Structures and FMaps/FSets
    "FMap": "Require Import FSets.FMapInterface.",
    "FSet": "Require Import FSets.FSetInterface.",
    "MSet": "Require Import MSets.MSetInterface.",
    
    # Mathematical theories
    "prime": "Require Import Znumtheory.",
    "Znumtheory": "Require Import Znumtheory.",
    "divisible": "Require Import Znumtheory.",
    "gcd": "Require Import Znumtheory.",
    "lcm": "Require Import Znumtheory.",
    "even": "Require Import Arith.",  # For natural numbers
    "odd": "Require Import Arith.",  # For natural numbers
    "Zeven": "Require Import ZArith.",  # For integers
    "Zodd": "Require Import ZArith.",  # For integers
}

# Define more specific scope requirements based on types
TYPE_TO_SCOPE = {
    "Z": "Z_scope",
    "R": "R_scope",
    "Q": "Q_scope",
    "N": "N_scope",
    "nat": "nat_scope",  # Default in Coq
    "Prop": "type_scope",
    "Set": "type_scope",
    "Type": "type_scope",
    "list": "list_scope",
    "string": "string_scope",
}

# Dictionary of modules that should be loaded together
MODULE_GROUPS = {
    "Reals": ["Rbasic_fun", "Rfunctions", "Rseries", "Rtrigo", "Ranalysis"],
    "ZArith": ["Znumtheory", "Zdiv", "Zpower", "Zabs"],
    "Lists": ["List", "ListSet", "ListTactics"],
    "Sets": ["Ensembles", "Finite_sets", "Powerset"],
    "Sorting": ["Sorted", "Permutation", "SetoidList"],
}

# Functions or tactics that require specific scope to be open
REQUIRES_SCOPE = {
    "ring": {
        "nat": "",  # Default scope
        "Z": "Z_scope",
        "R": "R_scope",
        "Q": "Q_scope",
        "N": "N_scope",
    },
    "field": {
        "R": "R_scope",
        "Q": "Q_scope",
    },
}

def detect_number_type(coq_code):
    """
    Analyze the Coq code to determine which number types are used.
    This helps in determining which scopes need to be opened.
    
    Args:
        coq_code (str): The Coq code to analyze
        
    Returns:
        list: List of detected number types ['Z', 'nat', 'R', etc.]
    """
    detected_types = []
    
    # Look for common type declarations

    if re.search(r':\s*Z\b', coq_code) or re.search(r'Z\.\w+', coq_code):
        detected_types.append('Z')
    if re.search(r':\s*nat\b', coq_code) or re.search(r'Nat\.\w+', coq_code):
        detected_types.append('nat')
    if re.search(r':\s*R\b', coq_code) or re.search(r'R\.\w+', coq_code):
        detected_types.append('R')
    if re.search(r':\s*Q\b', coq_code) or re.search(r'Q\.\w+', coq_code):
        detected_types.append('Q')
    if re.search(r':\s*N\b', coq_code) or re.search(r'N\.\w+', coq_code):
        detected_types.append('N')
    
    return detected_types

def get_ring_imports_for_type(number_type):
    """
    Return the appropriate ring-related imports for specific number types.
    Different number types require different Ring implementations.
    
    Args:
        number_type (str): The detected number type ('nat', 'Z', 'Q', 'R')
        
    Returns:
        list: List of required imports for ring operations with this type
    """
    if number_type == 'nat':
        return ["Require Import Arith.", "Require Import ArithRing."]
    elif number_type == 'Z':
        return ["Require Import ZArith.", "Require Import ZArithRing."]
    elif number_type == 'Q':
        return ["Require Import QArith.", "Require Import QArith_base.", "Require Import Qring."]
    elif number_type == 'R':
        return ["Require Import Reals.", "Require Import RingMicromega."]
    elif number_type == 'N':
        return ["Require Import NArith.", "Require Import NArithRing."]
    else:
        # Default to generic Ring
        return ["Require Import Ring."]

def analyze_coq_code(coq_code):
    """
    Enhanced analyze_coq_code function that handles ring imports properly.
    """
    import re
    
    # Lines that are already imported
    already_imported = set()
    for line in coq_code.split('\n'):
        if line.strip().startswith("Require Import"):
            already_imported.add(line.strip())
        if line.strip().startswith("Open Scope"):
            already_imported.add(line.strip())
    
    # Look for references that need imports
    missing_imports = set()
    missing_scopes = set()
    
    # Detect number types used in the code
    number_types = detect_number_type(coq_code)
    
    # Check for tactics and other references that need imports
    for reference, import_statement in COQ_IMPORTS.items():
        # Skip if import already exists or if no import needed
        if not import_statement or import_statement in already_imported:
            continue
        
        # Check for exact match
        if reference in coq_code:
            missing_imports.add(import_statement)
        
        # Check for references followed by a dot (e.g., Z.add)
        if reference.endswith(".") and re.search(fr'\b{re.escape(reference)}\w+', coq_code):
            missing_imports.add(import_statement)
    
    # Add corresponding scopes for detected number types
    for num_type in number_types:
        if num_type in TYPE_TO_SCOPE:
            scope = TYPE_TO_SCOPE[num_type]
            scope_directive = f"Open Scope {scope}."
            if scope_directive not in already_imported and scope not in ["nat_scope", "type_scope"]:
                missing_scopes.add(scope_directive)
    
    # Check for tactics that require specific scopes
    for tactic, scopes in REQUIRES_SCOPE.items():
        if tactic in coq_code:
            for num_type in number_types:
                if num_type in scopes and scopes[num_type]:
                    scope_directive = f"Open Scope {scopes[num_type]}."
                    if scope_directive not in already_imported:
                        missing_scopes.add(scope_directive)
    
    # Special handling for the ring tactic - we need to be careful which ring implementation to use
    if "ring" in coq_code.lower():
        # Remove the generic Ring import if we find a specific number type
        if "Require Import Ring." in missing_imports:
            missing_imports.remove("Require Import Ring.")
        
        # Add the appropriate ring implementation for our number types
        if number_types:
            # Primary type is the first one we detected - use that for ring
            primary_type = number_types[0]
            ring_imports = get_ring_imports_for_type(primary_type)
            for ring_import in ring_imports:
                if ring_import and ring_import not in already_imported:
                    missing_imports.add(ring_import)
        else:
            # No specific type detected, add the generic Ring import
            missing_imports.add("Require Import Ring.")
    
    # Ensure we don't mix ring implementations for different number types
    # If we have nat, remove QArith, and vice versa
    if 'nat' in number_types and "Require Import QArith." in missing_imports:
        missing_imports.remove("Require Import QArith.")
    if 'nat' in number_types and "Require Import Qring." in missing_imports:
        missing_imports.remove("Require Import Qring.")
    
    # Same for Z and Q
    if 'Z' in number_types and "Require Import QArith." in missing_imports:
        missing_imports.remove("Require Import QArith.")
    if 'Z' in number_types and "Require Import Qring." in missing_imports:
        missing_imports.remove("Require Import Qring.")
    
    return list(missing_imports), list(missing_scopes)


def add_required_imports(coq_code):
    """
    Add any missing imports to the Coq code.
    
    Args:
        coq_code (str): The generated Coq code
        
    Returns:
        str: The Coq code with necessary imports added
    """
    import re
    
    missing_imports, missing_scopes = analyze_coq_code(coq_code)
    
    if not missing_imports and not missing_scopes:
        return coq_code
        
    # Prepare imports in a predictable order
    missing_imports.sort()
    missing_scopes.sort()
    
    # Find the right place to add imports (beginning of file)
    lines = coq_code.split('\n')
    import_index = 0
    
    # Find the last import line if one exists
    for i, line in enumerate(lines):
        if line.strip().startswith("Require Import"):
            import_index = i + 1
    
    # Add missing imports after the last import
    for import_stmt in missing_imports:
        # Check if the import statement or a similar one already exists
        similar_exists = False
        for i, line in enumerate(lines):
            if line.strip() == import_stmt:
                similar_exists = True
                break
        
        if not similar_exists:
            lines.insert(import_index, import_stmt)
            import_index += 1
    
    # Add missing scopes after imports
    for scope_stmt in missing_scopes:
        # Check if the scope statement or a similar one already exists
        similar_exists = False
        for i, line in enumerate(lines):
            if line.strip() == scope_stmt:
                similar_exists = True
                break
        
        if not similar_exists:
            lines.insert(import_index, scope_stmt)
            import_index += 1
        
    # Add a blank line after imports if needed
    if import_index > 0 and import_index < len(lines) and lines[import_index].strip():
        lines.insert(import_index, "")
    
    return '\n'.join(lines)

# Special handling for ring tactic which is used frequently
def ensure_ring_import(coq_code):
    """
    Specifically ensure that the Ring module is imported if the ring tactic is used.
    
    Args:
        coq_code (str): The Coq code to check
        
    Returns:
        str: The Coq code with Ring import added if needed
    """
    if "ring" in coq_code.lower() and "Require Import Ring" not in coq_code:
        lines = coq_code.split('\n')
        import_index = 0
        
        # Find the last import
        for i, line in enumerate(lines):
            if line.strip().startswith("Require Import"):
                import_index = i + 1
        
        # Add ring import
        lines.insert(import_index, "Require Import Ring.")
        
        # Add a blank line if needed
        if import_index + 1 < len(lines) and lines[import_index + 1].strip():
            lines.insert(import_index + 1, "")
            
        return '\n'.join(lines)
    
    return coq_code

if __name__ == "__main__":
    # Test with a sample Coq code
    sample_code = """
Theorem example: forall n: Z, (exists k: Z, n = 2 * k) -> (exists m: Z, n * n = 2 * m).
Proof.
  intros n [k H].
  exists (2 * k * k).
  rewrite H.
  ring.
Qed.
"""
    
    improved_code = add_required_imports(sample_code)
    print("Original code:")
    print(sample_code)
    print("\nImproved code:")
    print(improved_code)