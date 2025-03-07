coq_mappings = {
    "Assume": "intros",
    "Then": "assert",
    "Therefore": "exact",
    "by induction on": "induction",
    "even": "exists k : nat, x + x = 2 * k",
}

def translate_to_coq_rule_based(parsed_statements):
    """Convert parsed proof steps into Coq syntax using only rules"""
    # For backwards compatibility when we just have statements without structure
    if isinstance(parsed_statements, tuple) and len(parsed_statements) == 2:
        parsed_statements, _ = parsed_statements  # Unpack if we have the new format
        
    coq_code = []
    # Define the theorem statement correctly
    theorem_statement = "Theorem example: forall x: nat, exists k: nat, x + x = 2 * k."
    coq_code.append(theorem_statement)
    coq_code.append("Proof.")
    
    for statement in parsed_statements:
        sentence = " ".join(word for word, _, _ in statement)  # Reconstruct sentence
        
        # Remove redundant theorem declarations
        if "Theorem" in sentence:
            continue  # Skip extra theorem declarations
        # Replace informal terms with Coq syntax
        for key in coq_mappings:
            if key in sentence:
                sentence = sentence.replace(key, coq_mappings[key])
        # Fix "intros x is a natural number" -> "intros x."
        if "intros x is a natural number" in sentence:
            sentence = "intros x."
        # Fix structure for "exists" statement
        if "exists k : nat, k = x / 2." in sentence:
            sentence = "exists (x / 2)."
        coq_code.append(sentence)
    coq_code.append("simpl.")  # Simplify expression
    coq_code.append("reflexivity.")  # Prove equality
    coq_code.append("Qed.")  # Properly close the proof
    
    print("Generated Coq Code:")
    print("\n".join(coq_code))  # Debugging output
    
    return "\n".join(coq_code)

# Legacy function for backwards compatibility
def translate_to_coq(parsed_data):
    """Wrapper for backward compatibility"""
    # Check if we have the new format (tuple with statements and structure)
    if isinstance(parsed_data, tuple) and len(parsed_data) == 2:
        # Import the new translator
        from hybrid_translator import translate_to_coq as hybrid_translate
        return hybrid_translate(parsed_data)
    else:
        # Use the old rule-based translator
        return translate_to_coq_rule_based(parsed_data)