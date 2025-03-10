"""
Lean formatter for translating IR proofs to Lean syntax.
Converts the intermediate representation to Lean proof scripts.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import re

from ir.proof_ir import ProofIR, ProofNode, Expression, NodeType, ExprType, TacticType, TacticInfo
from backends.backend_interface import ProverBackend

class LeanFormatter(ProverBackend):
    """
    Formatter for Lean theorem prover.
    Converts proof IR to Lean syntax.
    """
    
    def __init__(self):
        """Initialize the Lean formatter."""
        self.library_imports = {}
        self.concept_mappings = self._initialize_concept_mappings()
        self.tactic_mappings = self._initialize_tactic_mappings()
    
    @property
    def name(self) -> str:
        """Get the name of the prover."""
        return "Lean"
    
    @property
    def version(self) -> str:
        """Get the version of the prover."""
        try:
            return self._get_lean_version()
        except:
            return "Unknown"
    
    @property
    def supported_libraries(self) -> List[str]:
        """Get the supported libraries."""
        return ["mathlib", "core", "init", "data"]
    
    def translate(self, proof_ir: ProofIR) -> str:
        """
        Translate the proof IR to Lean syntax.
        
        Args:
            proof_ir: The intermediate representation of the proof
            
        Returns:
            The Lean proof script
        """
        # Start with necessary imports
        script_lines = self._generate_imports(proof_ir)
        script_lines.append("")
        
        # Generate the theorem statement
        theorem_statement = self._generate_theorem_statement(proof_ir.theorem, proof_ir.domain)
        script_lines.append(theorem_statement)
        
        # Generate the proof
        proof_lines = self._generate_proof(proof_ir.proof_tree, proof_ir.pattern, proof_ir.tactics)
        script_lines.extend(proof_lines)
        
        return "\n".join(script_lines)
    
    def verify(self, proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify the proof script with Lean.
        
        Args:
            proof_script: The proof script to verify
            filename: Optional filename to save the script to
            
        Returns:
            Tuple of (verification success, error message if any)
        """
        import subprocess
        import tempfile
        import os
        
        # Create a temporary file if no filename provided
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.lean', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(proof_script.encode())
            filename = temp_filename
        else:
            # Write to the specified file
            with open(filename, 'w') as f:
                f.write(proof_script)
        
        try:
            # Run Lean on the file
            result = subprocess.run(
                ["lean", filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Lean verification timed out."
        except FileNotFoundError:
            return False, "Lean executable not found. Please ensure Lean is installed."
        finally:
            # Clean up temporary file if we created one
            if filename == temp_filename:
                try:
                    os.remove(temp_filename)
                except:
                    pass
    
    def map_concept(self, concept: str, domain: Optional[str] = None) -> str:
        """
        Map a mathematical concept to its Lean representation.
        
        Args:
            concept: The concept to map
            domain: Optional domain for context
            
        Returns:
            The Lean representation
        """
        # Check for exact match
        if concept in self.concept_mappings:
            return self.concept_mappings[concept]
        
        # Check for domain-specific mappings
        if domain and domain in self.domain_specific_mappings:
            domain_mappings = self.domain_specific_mappings[domain]
            if concept in domain_mappings:
                return domain_mappings[concept]
        
        # No mapping found, return as-is
        return concept
    
    def map_tactic(self, tactic_type: TacticType, args: List[Any] = None) -> str:
        """
        Map a generic tactic to its Lean syntax.
        
        Args:
            tactic_type: The type of tactic
            args: Optional arguments for the tactic
            
        Returns:
            The Lean tactic syntax
        """
        args = args or []
        
        if tactic_type in self.tactic_mappings:
            tactic_template = self.tactic_mappings[tactic_type]
            
            # Replace placeholder with actual arguments
            if "{args}" in tactic_template:
                args_str = " ".join(str(arg) for arg in args)
                return tactic_template.replace("{args}", args_str)
            
            return tactic_template
        
        # Default case: unknown tactic
        return f"sorry -- Unknown tactic: {tactic_type.value}"
    
    def interactive_session(self, proof_script: str) -> str:
        """
        Start an interactive session with Lean.
        
        Args:
            proof_script: The proof script to load
            
        Returns:
            Path to a temporary file containing the proof script
        """
        import tempfile
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.lean', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(proof_script.encode())
        
        print(f"\nInteractive Lean Session:")
        print(f"1. A temporary file has been created at: {temp_filename}")
        print(f"2. You can run Lean in VS Code with this file to work interactively.")
        print(f"3. Use 'lean --make {temp_filename}' to check the proof from command line.")
        
        return temp_filename
    
    def get_file_extension(self) -> str:
        """
        Get the file extension for Lean scripts.
        
        Returns:
            The file extension
        """
        return ".lean"
    
    def _check_installation(self) -> None:
        """
        Check if Lean is installed on the system.
        Raises an exception if Lean is not installed.
        """
        import subprocess
        
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Lean is installed but returned non-zero exit code.")
        except FileNotFoundError:
            raise RuntimeError("Lean executable not found. Please ensure Lean is installed.")
    
    def _get_lean_version(self) -> str:
        """
        Get the installed Lean version.
        
        Returns:
            The Lean version string
        """
        import subprocess
        
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse version from output
                version_match = re.search(r'Lean \(version ([^)]+)\)', result.stdout)
                if version_match:
                    return version_match.group(1)
                return "Unknown"
            else:
                return "Unknown"
        except FileNotFoundError:
            return "Not installed"
    
    def _process_feedback_impl(self, error_message: str) -> Dict[str, Any]:
        """
        Process error feedback from Lean.
        
        Args:
            error_message: The error message from Lean
            
        Returns:
            Dictionary with structured error information
        """
        # Default structure
        error_info = {
            "type": "unknown",
            "line": None,
            "column": None,
            "message": error_message,
            "suggestion": None
        }
        
        # Parse line and column information
        line_match = re.search(r'file .*?:(\d+):(\d+):', error_message)
        if line_match:
            error_info["line"] = int(line_match.group(1))
            error_info["column"] = int(line_match.group(2))
        
        # Categorize error types
        if "unknown identifier" in error_message:
            error_info["type"] = "unknown_identifier"
            identifier_match = re.search(r'unknown identifier [\'"]([^\'"]+)[\'"]', error_message)
            if identifier_match:
                error_info["identifier"] = identifier_match.group(1)
                error_info["suggestion"] = f"Check the spelling of '{identifier_match.group(1)}' or import the relevant module."
        
        elif "type mismatch" in error_message:
            error_info["type"] = "type_mismatch"
            error_info["suggestion"] = "The types do not match. Check the type of each expression."
        
        elif "declaration has metavariables" in error_message:
            error_info["type"] = "incomplete_proof"
            error_info["suggestion"] = "The proof is incomplete. Try using more specific tactics or explicit arguments."
        
        elif "tactic failed" in error_message:
            error_info["type"] = "tactic_failure"
            error_info["suggestion"] = "The tactic failed. Try a different approach or more specific tactics."
        
        return error_info
    
    def _initialize_concept_mappings(self) -> Dict[str, str]:
        """
        Initialize mappings from mathematical concepts to Lean representations.
        
        Returns:
            Dictionary mapping concepts to Lean representations
        """
        return {
            # Basic number types
            "natural number": "ℕ",
            "integer": "ℤ",
            "rational": "ℚ",
            "real": "ℝ",
            "complex": "ℂ",
            
            # Set theory
            "set": "set",
            "subset": "⊆",
            "element": "∈",
            "union": "∪",
            "intersection": "∩",
            "empty set": "∅",
            
            # Logic
            "and": "∧",
            "or": "∨",
            "not": "¬",
            "implies": "→",
            "if and only if": "↔",
            "for all": "∀",
            "there exists": "∃",
            
            # Functions
            "function": "Function",
            "injective": "Function.Injective",
            "surjective": "Function.Surjective",
            "bijective": "Function.Bijective",
            
            # Algebra
            "group": "Group",
            "ring": "Ring",
            "field": "Field",
            
            # Arithmetic
            "divides": "∣",
            "prime": "Nat.Prime",
            "even": "Even",
            "odd": "Odd"
        }
    
    def _initialize_tactic_mappings(self) -> Dict[TacticType, str]:
        """
        Initialize mappings from tactic types to Lean tactic syntax.
        
        Returns:
            Dictionary mapping tactic types to Lean syntax
        """
        return {
            TacticType.INTRO: "intro {args}",
            TacticType.APPLY: "apply {args}",
            TacticType.REWRITE: "rw [{args}]",
            TacticType.CASE_ANALYSIS: "cases {args}",
            TacticType.INDUCTION: "induction {args}",
            TacticType.CONTRADICTION: "contradiction",
            TacticType.EXISTS: "use {args}",
            TacticType.SIMPLIFY: "simp",
            TacticType.AUTO: "tauto",
            TacticType.CUSTOM: "{args}"
        }
    
    def _generate_imports(self, proof_ir: ProofIR) -> List[str]:
        """
        Generate Lean import statements based on the proof domain.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            List of import statements
        """
        imports = ["import Mathlib.Tactic.Basic"]
        
        # Add domain-specific imports
        domain = proof_ir.domain.get("primary_domain", "")
        
        if domain in ["11", "12-20"]:  # Number theory or algebra
            imports.append("import Mathlib.Algebra.BigOperators")
            imports.append("import Mathlib.NumberTheory.Primes")
        
        elif domain in ["26-42"]:  # Analysis
            imports.append("import Mathlib.Analysis.RealFunction")
            
        elif domain in ["54-55"]:  # Topology
            imports.append("import Mathlib.Topology.Basic")
            
        # Add pattern-specific imports
        pattern = proof_ir.pattern.get("name", "")
        
        if pattern == "induction":
            imports.append("import Mathlib.Tactic.Induction")
        
        # Deduplicate and sort
        return sorted(list(set(imports)))
    
    def _generate_theorem_statement(self, theorem_node: ProofNode, domain_info: Dict[str, Any]) -> str:
        """
        Generate a Lean theorem statement from the theorem node.
        
        Args:
            theorem_node: The theorem node from the IR
            domain_info: Domain information for context
            
        Returns:
            Lean theorem statement
        """
        # Extract theorem statement
        if isinstance(theorem_node.content, str):
            theorem_text = theorem_node.content
        else:
            # If it's an Expression, convert to string
            theorem_text = str(theorem_node.content.value)
        
        # Convert common mathematical statements to Lean syntax
        theorem_text = self._convert_to_lean_syntax(theorem_text, domain_info)
        
        # Format as a Lean theorem
        return f"theorem example : {theorem_text} :="
    
    def _generate_proof(self, proof_tree: List[ProofNode], pattern_info: Dict[str, Any], 
                       tactics: List[TacticInfo]) -> List[str]:
        """
        Generate a Lean proof from the proof tree.
        
        Args:
            proof_tree: The proof tree from the IR
            pattern_info: Pattern information
            tactics: Suggested tactics
            
        Returns:
            List of lines in the Lean proof
        """
        lines = ["by {"]
        
        # Get the pattern name
        pattern_name = pattern_info.get("name", "direct")
        
        # Choose proof strategy based on pattern
        if pattern_name == "induction":
            lines.extend(self._generate_induction_proof(proof_tree, tactics))
        elif pattern_name == "contradiction":
            lines.extend(self._generate_contradiction_proof(proof_tree, tactics))
        elif pattern_name == "cases":
            lines.extend(self._generate_cases_proof(proof_tree, tactics))
        else:
            # Default to a linear proof
            lines.extend(self._generate_linear_proof(proof_tree, tactics))
        
        # If proof is empty or very simple, add a default tactic
        if len(lines) <= 2:
            lines.append("  sorry -- Fill in the proof")
        
        lines.append("}")
        return lines
    
    def _generate_induction_proof(self, proof_tree: List[ProofNode], 
                                 tactics: List[TacticInfo]) -> List[str]:
        """
        Generate a Lean induction proof.
        
        Args:
            proof_tree: The proof tree
            tactics: Suggested tactics
            
        Returns:
            List of lines in the Lean proof
        """
        lines = []
        
        # Find induction variable from tactics
        induction_var = None
        for tactic in tactics:
            if tactic.tactic_type == TacticType.INDUCTION and tactic.arguments:
                induction_var = tactic.arguments[0]
                break
        
        # Default to 'n' if no variable found
        if not induction_var:
            induction_var = "n"
        
        # Add induction tactic
        lines.append(f"  induction {induction_var}")
        
        # Find base case and inductive step in the proof tree
        base_case = None
        inductive_step = None
        
        for node in proof_tree:
            if node.node_type == NodeType.INDUCTION_BASE:
                base_case = node
            elif node.node_type == NodeType.INDUCTION_STEP:
                inductive_step = node
        
        # Generate base case
        lines.append("  case zero => {")
        if base_case:
            lines.append(f"    -- {base_case.content}")
            lines.append("    simp")
        else:
            lines.append("    simp")
        lines.append("  }")
        
        # Generate inductive step
        lines.append("  case succ => {")
        if inductive_step:
            lines.append(f"    -- {inductive_step.content}")
            lines.append("    rw [ih]")
            lines.append("    ring")
        else:
            lines.append("    rw [ih]")
            lines.append("    ring")
        lines.append("  }")
        
        return lines
    
    def _generate_contradiction_proof(self, proof_tree: List[ProofNode], 
                                     tactics: List[TacticInfo]) -> List[str]:
        """
        Generate a Lean contradiction proof.
        
        Args:
            proof_tree: The proof tree
            tactics: Suggested tactics
            
        Returns:
            List of lines in the Lean proof
        """
        lines = []
        
        # Add contradiction tactic
        lines.append("  by_contra h")
        
        # Find the contradiction assumption
        assumption = None
        for node in proof_tree:
            if node.node_type == NodeType.ASSUMPTION and node.metadata.get("contradiction_assumption"):
                assumption = node
                break
        
        # If we found an explicit contradiction assumption, add it
        if assumption:
            lines.append(f"  -- {assumption.content}")
        
        # Add steps to derive contradiction
        contradiction_found = False
        for node in proof_tree:
            if node.node_type == NodeType.CONTRADICTION:
                lines.append(f"  -- {node.content}")
                contradiction_found = True
        
        # Ensure we have a contradiction
        if not contradiction_found:
            lines.append("  -- Derive contradiction")
            lines.append("  exfalso")
        
        return lines
    
    def _generate_cases_proof(self, proof_tree: List[ProofNode], 
                             tactics: List[TacticInfo]) -> List[str]:
        """
        Generate a Lean cases proof.
        
        Args:
            proof_tree: The proof tree
            tactics: Suggested tactics
            
        Returns:
            List of lines in the Lean proof
        """
        lines = []
        
        # Find case variable from tactics
        case_var = None
        for tactic in tactics:
            if tactic.tactic_type == TacticType.CASE_ANALYSIS and tactic.arguments:
                case_var = tactic.arguments[0]
                break
        
        # Default to 'h' if no variable found
        if not case_var:
            case_var = "h"
        
        # Add cases tactic
        lines.append(f"  cases {case_var}")
        
        # Process each case
        case_nodes = [node for node in proof_tree if node.node_type == NodeType.CASE]
        
        if case_nodes:
            for i, case_node in enumerate(case_nodes):
                case_num = case_node.metadata.get("case_number", str(i + 1))
                lines.append(f"  case {case_num} => {{")
                lines.append(f"    -- {case_node.content}")
                
                # Add child steps
                for child in case_node.children:
                    lines.append(f"    -- {child.content}")
                
                lines.append("    simp")
                lines.append("  }")
        else:
            # Default cases if none explicitly found
            lines.append("  case left => {")
            lines.append("    simp")
            lines.append("  }")
            lines.append("  case right => {")
            lines.append("    simp")
            lines.append("  }")
        
        return lines
    
    def _generate_linear_proof(self, proof_tree: List[ProofNode], 
                              tactics: List[TacticInfo]) -> List[str]:
        """
        Generate a linear Lean proof.
        
        Args:
            proof_tree: The proof tree
            tactics: Suggested tactics
            
        Returns:
            List of lines in the Lean proof
        """
        lines = []
        
        # Process assumptions
        assumptions = [node for node in proof_tree if node.node_type == NodeType.ASSUMPTION]
        if assumptions:
            for assumption in assumptions:
                # Extract variables from assumption
                var_match = re.search(r'let\s+([a-zA-Z][a-zA-Z0-9]*)\b|assume\s+([a-zA-Z][a-zA-Z0-9]*)\b', 
                                     assumption.content, re.IGNORECASE)
                var = var_match.group(1) or var_match.group(2) if var_match else None
                
                if var:
                    lines.append(f"  intro {var}")
                else:
                    lines.append("  intro h")
                
                lines.append(f"  -- {assumption.content}")
        else:
            # Add intro tactic if suggested
            for tactic in tactics:
                if tactic.tactic_type == TacticType.INTRO:
                    if tactic.arguments:
                        lines.append(f"  intro {' '.join(tactic.arguments)}")
                    else:
                        lines.append("  intro h")
                    break
        
        # Process steps
        steps = [node for node in proof_tree if node.node_type == NodeType.STEP]
        if steps:
            for step in steps:
                lines.append(f"  -- {step.content}")
        
        # Process conclusion
        conclusions = [node for node in proof_tree if node.node_type == NodeType.CONCLUSION]
        if conclusions:
            for conclusion in conclusions:
                lines.append(f"  -- {conclusion.content}")
        
        # Add default tactics if no steps
        if not steps and not conclusions:
            lines.append("  simp")
            
            # Add suggestions from tactics
            for tactic in tactics:
                if tactic.tactic_type not in [TacticType.INTRO, TacticType.AUTO]:
                    lines.append(f"  {self.map_tactic(tactic.tactic_type, tactic.arguments)}")
        
        return lines
    
    def _convert_to_lean_syntax(self, text: str, domain_info: Dict[str, Any]) -> str:
        """
        Convert mathematical text to Lean syntax.
        
        Args:
            text: The mathematical text
            domain_info: Domain information for context
            
        Returns:
            Lean syntax
        """
        # Replace common math phrases with Lean syntax
        replacements = [
            (r'\bfor\s+all\s+([a-zA-Z][a-zA-Z0-9]*)\b', r'∀ \1,'),
            (r'\bthere\s+exists\s+([a-zA-Z][a-zA-Z0-9]*)\b', r'∃ \1,'),
            (r'\b([a-zA-Z][a-zA-Z0-9]*)\s+is\s+a\s+natural\s+number\b', r'\1 : ℕ'),
            (r'\b([a-zA-Z][a-zA-Z0-9]*)\s+is\s+an?\s+integer\b', r'\1 : ℤ'),
            (r'\b([a-zA-Z][a-zA-Z0-9]*)\s+is\s+even\b', r'Even \1'),
            (r'\b([a-zA-Z][a-zA-Z0-9]*)\s+is\s+odd\b', r'Odd \1'),
            (r'\b([a-zA-Z][a-zA-Z0-9]*)\s*\+\s*([a-zA-Z][a-zA-Z0-9]*)\s+is\s+even\b', r'Even (\1 + \2)'),
            (r'\bif\s+([^,]+),\s+then\s+([^.]+)', r'(\1) → (\2)'),
            (r'\band\b', r'∧'),
            (r'\bor\b', r'∨'),
            (r'\bnot\b', r'¬'),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Special case for "x + x is even"
        if re.search(r'\b([a-zA-Z])\s*\+\s*\1\s+is\s+even\b', result, re.IGNORECASE):
            var = re.search(r'\b([a-zA-Z])\s*\+\s*\1\s+is\s+even\b', result, re.IGNORECASE).group(1)
            result = f"∃ k : ℕ, {var} + {var} = 2 * k"
        
        return result


# Register the backend
from backends.backend_interface import BackendRegistry
BackendRegistry.register("lean", LeanFormatter)