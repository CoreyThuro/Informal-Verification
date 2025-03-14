"""
Feedback loop system for verification results.
Learns from verification results to improve future translations.
"""

import re
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger("feedback_loop")

class ErrorAnalyzer:
    """
    Analyzer for theorem prover errors.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the error analyzer.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Load error patterns for different provers.
        
        Returns:
            Dictionary of error patterns
        """
        # These could be loaded from a file in a real implementation
        return {
            "coq": {
                "undefined_reference": {
                    "pattern": r'The reference ([^ ]+) was not found',
                    "extraction": lambda m: {"reference": m.group(1)},
                    "suggestion": "Check if the reference is defined or imported"
                },
                "type_mismatch": {
                    "pattern": r'Unable to unify "([^"]+)" with "([^"]+)"',
                    "extraction": lambda m: {"expected": m.group(1), "actual": m.group(2)},
                    "suggestion": "The types do not match. Check the types of expressions."
                },
                "syntax_error": {
                    "pattern": r'Syntax error',
                    "extraction": lambda m: {},
                    "suggestion": "Check for syntax errors in the proof"
                },
                "invalid_pattern_matching": {
                    "pattern": r'Invalid pattern-matching',
                    "extraction": lambda m: {},
                    "suggestion": "Check the pattern matching syntax"
                },
                "incomplete_proof": {
                    "pattern": r'Attempt to save an incomplete proof',
                    "extraction": lambda m: {},
                    "suggestion": "The proof is incomplete. There are still goals to be proven."
                }
            },
            "lean": {
                "unknown_identifier": {
                    "pattern": r'unknown identifier [\'"]([^\'"]+)[\'"]',
                    "extraction": lambda m: {"identifier": m.group(1)},
                    "suggestion": "Check if the identifier is defined or imported"
                },
                "type_mismatch": {
                    "pattern": r'type mismatch at application[^\n]*\nfunction[^\n]*\nargument[^\n]*has type',
                    "extraction": lambda m: {},
                    "suggestion": "The types do not match. Check the types of expressions."
                },
                "tactic_failure": {
                    "pattern": r'tactic [\'"]([^\'"]+)[\'"] failed',
                    "extraction": lambda m: {"tactic": m.group(1)},
                    "suggestion": "The tactic failed. Try a different approach or more specific tactics."
                },
                "declaration_has_metavariables": {
                    "pattern": r'declaration has metavariables',
                    "extraction": lambda m: {},
                    "suggestion": "The proof is incomplete. There are still goals to be solved."
                }
            }
        }
    
    def analyze_error(self, error_message: str, proof_script: str, domain: str, prover: str) -> Dict[str, Any]:
        """
        Analyze an error from a theorem prover.
        
        Args:
            error_message: The error message from the prover
            proof_script: The proof script that failed verification
            domain: The mathematical domain
            prover: The theorem prover
            
        Returns:
            Dictionary with error analysis
        """
        # Match against known error patterns
        matched_patterns = self._match_error_patterns(error_message, prover)
        if not matched_patterns:
            return self._generic_error_analysis(error_message)
        
        # Get the best matching pattern
        best_match = max(matched_patterns, key=lambda x: x["score"])
        
        # Generate correction based on the error type
        if best_match["type"] == "undefined_reference":
            return self._handle_undefined_reference(best_match, proof_script, domain, prover)
        elif best_match["type"] == "type_mismatch":
            return self._handle_type_mismatch(best_match, proof_script, domain)
        elif best_match["type"] == "incomplete_proof":
            return self._handle_incomplete_proof(best_match, proof_script)
        elif best_match["type"] == "syntax_error":
            return self._handle_syntax_error(best_match, proof_script)
        
        # Default handler
        return {
            "error_type": best_match["type"],
            "message": best_match.get("message", "Unknown error"),
            "suggestion": best_match.get("suggestion", "Check the error message for details.")
        }
    
    def _match_error_patterns(self, error_message: str, prover: str) -> List[Dict[str, Any]]:
        """
        Match the error message against known patterns.
        
        Args:
            error_message: The error message
            prover: The theorem prover
            
        Returns:
            List of matched patterns with scores
        """
        matched_patterns = []
        
        if prover not in self.error_patterns:
            return matched_patterns
        
        for error_type, pattern_info in self.error_patterns[prover].items():
            pattern = pattern_info["pattern"]
            match = re.search(pattern, error_message, re.IGNORECASE)
            
            if match:
                # Extract information from the match
                extracted_info = pattern_info["extraction"](match) if callable(pattern_info["extraction"]) else {}
                
                # Calculate score based on match length
                score = len(match.group(0)) / len(error_message)
                
                matched_patterns.append({
                    "type": error_type,
                    "score": score,
                    "message": match.group(0),
                    "suggestion": pattern_info.get("suggestion", ""),
                    "extracted_info": extracted_info
                })
        
        return matched_patterns
    
    def _generic_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """
        Generic error analysis for unrecognized errors.
        
        Args:
            error_message: The error message
            
        Returns:
            Dictionary with basic error analysis
        """
        # Extract line number if available
        line_match = re.search(r'line (\d+)', error_message)
        line_number = int(line_match.group(1)) if line_match else None
        
        return {
            "error_type": "unknown",
            "message": error_message,
            "line_number": line_number,
            "suggestion": "Check the error message for details."
        }
    
    def _handle_undefined_reference(self, error_match: Dict[str, Any], proof_script: str, 
                                  domain: str, prover: str) -> Dict[str, Any]:
        """
        Handle undefined reference errors.
        
        Args:
            error_match: The error match information
            proof_script: The proof script
            domain: The mathematical domain
            prover: The theorem prover
            
        Returns:
            Dictionary with error analysis and fix suggestion
        """
        # Extract the undefined reference
        reference = error_match.get("extracted_info", {}).get("reference")
        if not reference:
            return error_match
        
        # Check if the knowledge base is available
        if not self.kb:
            return {
                "error_type": "undefined_reference",
                "message": f"The reference '{reference}' was not found",
                "suggestion": f"Check if '{reference}' is defined or imported correctly."
            }
        
        # Look up the reference in the knowledge base
        libraries = self.kb.get_libraries_for_concept(reference, domain, prover)
        
        if libraries:
            # Check if required libraries are already imported
            missing_libraries = []
            for library in libraries:
                library_import = self._get_import_statement(library, prover)
                if library_import and library_import not in proof_script:
                    missing_libraries.append(library_import)
            
            if missing_libraries:
                # Suggest adding missing imports
                return {
                    "error_type": "missing_import",
                    "message": f"The reference '{reference}' requires importing additional libraries",
                    "suggestion": f"Add the following imports: {', '.join(missing_libraries)}",
                    "fix": {
                        "type": "add_import",
                        "imports": missing_libraries
                    }
                }
        
        # Default response if no specific fix found
        return {
            "error_type": "undefined_reference",
            "message": f"The reference '{reference}' was not found",
            "suggestion": f"Check if '{reference}' is defined or imported correctly."
        }
    
    def _handle_type_mismatch(self, error_match: Dict[str, Any], proof_script: str, 
                           domain: str) -> Dict[str, Any]:
        """
        Handle type mismatch errors.
        
        Args:
            error_match: The error match information
            proof_script: The proof script
            domain: The mathematical domain
            
        Returns:
            Dictionary with error analysis and fix suggestion
        """
        expected_type = error_match.get("extracted_info", {}).get("expected")
        actual_type = error_match.get("extracted_info", {}).get("actual")
        
        suggestion = "The types do not match"
        if expected_type and actual_type:
            suggestion += f": expected '{expected_type}' but got '{actual_type}'"
        
        return {
            "error_type": "type_mismatch",
            "message": error_match.get("message", "Type mismatch detected"),
            "suggestion": suggestion,
            "fix": {
                "type": "type_conversion",
                "expected": expected_type,
                "actual": actual_type
            } if expected_type and actual_type else None
        }
    
    def _handle_incomplete_proof(self, error_match: Dict[str, Any], proof_script: str) -> Dict[str, Any]:
        """
        Handle incomplete proof errors.
        
        Args:
            error_match: The error match information
            proof_script: The proof script
            
        Returns:
            Dictionary with error analysis and fix suggestion
        """
        # Find Qed. in the proof script
        qed_match = re.search(r'Qed\.', proof_script)
        if qed_match:
            # Suggest replacing Qed with Admitted
            return {
                "error_type": "incomplete_proof",
                "message": error_match.get("message", "Incomplete proof"),
                "suggestion": "The proof is incomplete. Try replacing 'Qed.' with 'Admitted.' to accept the proof as an axiom.",
                "fix": {
                    "type": "replace_qed",
                    "replace_with": "Admitted."
                }
            }
        
        return {
            "error_type": "incomplete_proof",
            "message": error_match.get("message", "Incomplete proof"),
            "suggestion": "The proof is incomplete. There are still goals to be proven."
        }
    
    def _handle_syntax_error(self, error_match: Dict[str, Any], proof_script: str) -> Dict[str, Any]:
        """
        Handle syntax errors.
        
        Args:
            error_match: The error match information
            proof_script: The proof script
            
        Returns:
            Dictionary with error analysis and fix suggestion
        """
        # Look for common syntax issues
        missing_dot = not re.search(r'\.\s*$', proof_script.strip())
        missing_qed = not re.search(r'Qed\.\s*$', proof_script.strip())
        
        if missing_qed:
            return {
                "error_type": "syntax_error",
                "message": error_match.get("message", "Syntax error"),
                "suggestion": "The proof might be missing a 'Qed.' at the end.",
                "fix": {
                    "type": "add_qed"
                }
            }
        elif missing_dot:
            return {
                "error_type": "syntax_error",
                "message": error_match.get("message", "Syntax error"),
                "suggestion": "Check for missing periods after tactics.",
                "fix": {
                    "type": "check_missing_periods"
                }
            }
        
        return {
            "error_type": "syntax_error",
            "message": error_match.get("message", "Syntax error"),
            "suggestion": "Check for syntax errors in the proof."
        }
    
    def _get_import_statement(self, library: str, prover: str) -> Optional[str]:
        """
        Get the import statement for a library.
        
        Args:
            library: The library name
            prover: The theorem prover
            
        Returns:
            The import statement or None if not available
        """
        if prover == "coq":
            return f"Require Import {library}."
        elif prover == "lean":
            return f"import {library}"
        return None


class FeedbackLoopSystem:
    """
    System for providing feedback from verification to translation.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the feedback loop system.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
        self.error_analyzer = ErrorAnalyzer(kb)
        self.feedback_history = {}
        self.feedback_database_path = os.path.join(
            os.path.dirname(__file__), 'data', 'feedback_history.json')
        
        # Load previous feedback if available
        self._load_feedback_history()
    
    def _load_feedback_history(self) -> None:
        """Load feedback history from file if available."""
        try:
            if os.path.exists(self.feedback_database_path):
                with open(self.feedback_database_path, 'r') as f:
                    self.feedback_history = json.load(f)
                logger.info(f"Loaded feedback history with {len(self.feedback_history)} entries")
        except Exception as e:
            logger.warning(f"Failed to load feedback history: {e}")
    
    def _save_feedback_history(self) -> None:
        """Save feedback history to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.feedback_database_path), exist_ok=True)
            
            with open(self.feedback_database_path, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
            logger.info(f"Saved feedback history with {len(self.feedback_history)} entries")
        except Exception as e:
            logger.warning(f"Failed to save feedback history: {e}")
    
    def process_verification_result(self, proof_ir: Any, formal_proof: str,
                                   verification_result: bool, error_message: Optional[str],
                                   target_prover: str) -> Dict[str, Any]:
        """
        Process a verification result and generate feedback.
        
        Args:
            proof_ir: The proof intermediate representation
            formal_proof: The formal proof script
            verification_result: Whether verification succeeded
            error_message: The error message if verification failed
            target_prover: The target theorem prover
            
        Returns:
            Dictionary with feedback information
        """
        # If verification succeeded, store successful translation for future reference
        if verification_result:
            self._store_successful_translation(proof_ir, formal_proof, target_prover)
            return {"success": True}
        
        # Analyze error
        domain = proof_ir.domain.get("primary_domain", "") if hasattr(proof_ir, "domain") else ""
        error_analysis = self.error_analyzer.analyze_error(
            error_message or "", formal_proof, domain, target_prover)
        
        # Generate feedback
        feedback = self._generate_feedback(error_analysis, proof_ir)
        
        # Store feedback for future reference
        self._store_feedback(proof_ir, feedback, target_prover)
        
        return {
            "success": False,
            "error_analysis": error_analysis,
            "feedback": feedback
        }
    
    def _generate_feedback(self, error_analysis: Dict[str, Any], proof_ir: Any) -> Dict[str, Any]:
        """
        Generate feedback based on error analysis.
        
        Args:
            error_analysis: The error analysis
            proof_ir: The proof intermediate representation
            
        Returns:
            Dictionary with feedback
        """
        error_type = error_analysis.get("error_type", "unknown")
        
        if error_type == "missing_import":
            return {
                "type": "missing_import",
                "message": error_analysis.get("message", "Missing import detected."),
                "suggestion": error_analysis.get("suggestion", "Add the required import."),
                "fix": error_analysis.get("fix", {})
            }
        
        elif error_type == "type_mismatch":
            return {
                "type": "type_mismatch",
                "message": error_analysis.get("message", "Type mismatch detected."),
                "suggestion": error_analysis.get("suggestion", "Check the types of expressions."),
                "fix": error_analysis.get("fix", {})
            }
        
        elif error_type == "incomplete_proof":
            return {
                "type": "incomplete_proof",
                "message": error_analysis.get("message", "Incomplete proof detected."),
                "suggestion": error_analysis.get("suggestion", "The proof is incomplete."),
                "fix": error_analysis.get("fix", {})
            }
        
        elif error_type == "syntax_error":
            return {
                "type": "syntax_error",
                "message": error_analysis.get("message", "Syntax error detected."),
                "suggestion": error_analysis.get("suggestion", "Check the syntax in the proof."),
                "fix": error_analysis.get("fix", {})
            }
        
        # Default feedback
        return {
            "type": "general",
            "message": "Verification failed.",
            "suggestion": error_analysis.get("suggestion", "Check the error message for details.")
        }
    
    def _store_successful_translation(self, proof_ir: Any, formal_proof: str, target_prover: str) -> None:
        """
        Store a successful translation for future reference.
        
        Args:
            proof_ir: The proof intermediate representation
            formal_proof: The formal proof script
            target_prover: The target theorem prover
        """
        try:
            # Create a hash to identify the proof IR
            proof_hash = self._hash_proof_ir(proof_ir)
            
            # Store in feedback history
            if proof_hash not in self.feedback_history:
                self.feedback_history[proof_hash] = {}
            
            self.feedback_history[proof_hash][target_prover] = {
                "success": True,
                "formal_proof": formal_proof,
                "domain": proof_ir.domain.get("primary_domain", "") if hasattr(proof_ir, "domain") else "",
                "pattern": proof_ir.pattern.get("name", "") if hasattr(proof_ir, "pattern") else "",
                "timestamp": self._get_timestamp()
            }
            
            # Save feedback history
            self._save_feedback_history()
        except Exception as e:
            logger.warning(f"Failed to store successful translation: {e}")
    
    def _store_feedback(self, proof_ir: Any, feedback: Dict[str, Any], target_prover: str) -> None:
        """
        Store feedback for future reference.
        
        Args:
            proof_ir: The proof intermediate representation
            feedback: The feedback information
            target_prover: The target theorem prover
        """
        try:
            # Create a hash to identify the proof IR
            proof_hash = self._hash_proof_ir(proof_ir)
            
            # Store in feedback history
            if proof_hash not in self.feedback_history:
                self.feedback_history[proof_hash] = {}
            
            # If there's no entry for this prover or it was previously unsuccessful
            if target_prover not in self.feedback_history[proof_hash] or \
               not self.feedback_history[proof_hash][target_prover].get("success", False):
                
                self.feedback_history[proof_hash][target_prover] = {
                    "success": False,
                    "feedback": feedback,
                    "domain": proof_ir.domain.get("primary_domain", "") if hasattr(proof_ir, "domain") else "",
                    "pattern": proof_ir.pattern.get("name", "") if hasattr(proof_ir, "pattern") else "",
                    "timestamp": self._get_timestamp()
                }
                
                # Save feedback history
                self._save_feedback_history()
        except Exception as e:
            logger.warning(f"Failed to store feedback: {e}")
    
    def get_similar_feedback(self, proof_ir: Any, target_prover: str) -> Optional[Dict[str, Any]]:
        """
        Get feedback for similar proofs.
        
        Args:
            proof_ir: The proof intermediate representation
            target_prover: The target theorem prover
            
        Returns:
            Feedback for similar proofs or None if not found
        """
        try:
            # Check for exact match first
            proof_hash = self._hash_proof_ir(proof_ir)
            if proof_hash in self.feedback_history and \
               target_prover in self.feedback_history[proof_hash]:
                return self.feedback_history[proof_hash][target_prover]
            
            # Then look for similar proofs
            domain = proof_ir.domain.get("primary_domain", "") if hasattr(proof_ir, "domain") else ""
            pattern = proof_ir.pattern.get("name", "") if hasattr(proof_ir, "pattern") else ""
            
            similar_feedback = []
            for _, prover_feedback in self.feedback_history.items():
                if target_prover in prover_feedback:
                    feedback_entry = prover_feedback[target_prover]
                    if feedback_entry.get("domain") == domain and feedback_entry.get("pattern") == pattern:
                        similar_feedback.append(feedback_entry)
            
            # If we found similar feedback, return the most recent successful one
            successful_feedback = [f for f in similar_feedback if f.get("success", False)]
            if successful_feedback:
                # Sort by timestamp (most recent first)
                successful_feedback.sort(key=lambda f: f.get("timestamp", 0), reverse=True)
                return successful_feedback[0]
            
            # If no successful feedback, return the most recent feedback
            if similar_feedback:
                similar_feedback.sort(key=lambda f: f.get("timestamp", 0), reverse=True)
                return similar_feedback[0]
        except Exception as e:
            logger.warning(f"Failed to get similar feedback: {e}")
        
        return None
    
    def apply_feedback_fixes(self, formal_proof: str, feedback: Dict[str, Any]) -> str:
        """
        Apply fixes from feedback to the formal proof.
        
        Args:
            formal_proof: The formal proof script
            feedback: The feedback information
            
        Returns:
            Modified formal proof with fixes applied
        """
        if not feedback or "fix" not in feedback:
            return formal_proof
        
        fix = feedback["fix"]
        fix_type = fix.get("type", "")
        
        try:
            if fix_type == "add_import":
                return self._add_imports(formal_proof, fix.get("imports", []))
            
            elif fix_type == "replace_qed":
                return self._replace_qed(formal_proof, fix.get("replace_with", "Admitted."))
            
            elif fix_type == "add_qed":
                return self._add_qed(formal_proof)
            
            elif fix_type == "check_missing_periods":
                return self._add_missing_periods(formal_proof)
        except Exception as e:
            logger.warning(f"Failed to apply feedback fix: {e}")
        
        return formal_proof
    
    def _add_imports(self, formal_proof: str, imports: List[str]) -> str:
        """
        Add import statements to the proof.
        
        Args:
            formal_proof: The formal proof script
            imports: List of import statements to add
            
        Returns:
            Modified formal proof with imports added
        """
        if not imports:
            return formal_proof
        
        # Find the end of existing imports
        import_end = 0
        lines = formal_proof.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(('Require Import', 'import')):
                import_end = i + 1
        
        # Insert new imports after existing ones
        result_lines = lines[:import_end]
        for import_stmt in imports:
            if import_stmt not in formal_proof:
                result_lines.append(import_stmt)
        result_lines.extend(lines[import_end:])
        
        return '\n'.join(result_lines)
    
    def _replace_qed(self, formal_proof: str, replace_with: str) -> str:
        """
        Replace Qed. with Admitted. or another terminator.
        
        Args:
            formal_proof: The formal proof script
            replace_with: The replacement for Qed.
            
        Returns:
            Modified formal proof
        """
        return re.sub(r'Qed\.', replace_with, formal_proof)
    
    def _add_qed(self, formal_proof: str) -> str:
        """
        Add Qed. to the end of the proof if missing.
        
        Args:
            formal_proof: The formal proof script
            
        Returns:
            Modified formal proof
        """
        if not re.search(r'Qed\.\s*$', formal_proof.strip()):
            return formal_proof.rstrip() + "\nQed."
        return formal_proof
    
    def _add_missing_periods(self, formal_proof: str) -> str:
        """
        Add missing periods after tactics.
        
        Args:
            formal_proof: The formal proof script
            
        Returns:
            Modified formal proof
        """
        lines = formal_proof.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Check if line is a tactic without a period
            if stripped and not stripped.endswith(('.', '{', '}', ')', '(*')):
                # Check if it looks like a tactic
                if re.match(r'^\s*[a-zA-Z]+', stripped):
                    line = line + "."
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _hash_proof_ir(self, proof_ir: Any) -> str:
        """
        Create a hash to identify a proof IR.
        
        Args:
            proof_ir: The proof intermediate representation
            
        Returns:
            Hash string
        """
        # Extract key information from proof IR
        properties = []
        
        if hasattr(proof_ir, "original_theorem"):
            properties.append(str(proof_ir.original_theorem))
        
        if hasattr(proof_ir, "original_proof"):
            properties.append(str(proof_ir.original_proof))
        
        if hasattr(proof_ir, "domain") and isinstance(proof_ir.domain, dict):
            properties.append(str(proof_ir.domain.get("primary_domain", "")))
        
        if hasattr(proof_ir, "pattern") and isinstance(proof_ir.pattern, dict):
            properties.append(str(proof_ir.pattern.get("name", "")))
        
        # Create a simple hash
        import hashlib
        properties_str = "::".join(properties)
        return hashlib.md5(properties_str.encode()).hexdigest()
    
    def _get_timestamp(self) -> int:
        """
        Get current timestamp.
        
        Returns:
            Current timestamp in seconds
        """
        import time
        return int(time.time())


# Singleton instance
feedback_system = FeedbackLoopSystem()

def process_verification_result(proof_ir: Any, formal_proof: str, 
                               verification_result: bool, error_message: Optional[str],
                               target_prover: str) -> Dict[str, Any]:
    """
    Process a verification result using the global feedback system.
    
    Args:
        proof_ir: The proof intermediate representation
        formal_proof: The formal proof script
        verification_result: Whether verification succeeded
        error_message: The error message if verification failed
        target_prover: The target theorem prover
        
    Returns:
        Dictionary with feedback information
    """
    return feedback_system.process_verification_result(
        proof_ir, formal_proof, verification_result, error_message, target_prover)

def apply_feedback_fixes(formal_proof: str, feedback: Dict[str, Any]) -> str:
    """
    Apply fixes from feedback to the formal proof.
    
    Args:
        formal_proof: The formal proof script
        feedback: The feedback information
        
    Returns:
        Modified formal proof with fixes applied
    """
    return feedback_system.apply_feedback_fixes(formal_proof, feedback)

def get_similar_feedback(proof_ir: Any, target_prover: str) -> Optional[Dict[str, Any]]:
    """
    Get feedback for similar proofs.
    
    Args:
        proof_ir: The proof intermediate representation
        target_prover: The target theorem prover
        
    Returns:
        Feedback for similar proofs or None if not found
    """
    return feedback_system.get_similar_feedback(proof_ir, target_prover)