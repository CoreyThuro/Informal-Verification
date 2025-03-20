"""
Update knowledge_base/simple_kb.py to enhance the integration with patterns and tactics
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
from ir.proof_ir import TacticType, TacticInfo

# Configure logging
logger = logging.getLogger(__name__)

class SimpleKnowledgeBase:
    """A simplified knowledge base that loads JSON data files directly."""
    
    def __init__(self, data_dir="knowledge_base/data"):
        """
        Initialize the simple knowledge base.
        
        Args:
            data_dir: Directory containing knowledge base data files
        """
        self.data_dir = data_dir
        self.concepts = self._load_concepts()
        self.libraries = self._load_libraries()
        self.patterns = self._load_patterns()
        self.msc_categories = self._load_msc_categories()
        logger.info(f"Initialized knowledge base with {len(self.concepts)} concepts, {len(self.patterns)} patterns")
        
    def _load_concepts(self) -> Dict[str, Any]:
        """
        Load concept definitions from JSON files.
        
        Returns:
            Dictionary of concept information
        """
        concepts = {}
        concepts_dir = os.path.join(self.data_dir, "concepts")
        
        if os.path.exists(concepts_dir):
            for filename in os.listdir(concepts_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(concepts_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            domain_concepts = json.load(f)
                            concepts.update(domain_concepts)
                        logger.debug(f"Loaded {len(domain_concepts)} concepts from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"Concepts directory not found: {concepts_dir}")
        
        return concepts
    
    def _load_libraries(self) -> Dict[str, Dict[str, Any]]:
        """
        Load library information from JSON files.
        
        Returns:
            Dictionary mapping provers to library information
        """
        libraries = {"coq": {}, "lean": {}}
        libraries_dir = os.path.join(self.data_dir, "libraries")
        
        if os.path.exists(libraries_dir):
            for filename in os.listdir(libraries_dir):
                if filename.endswith(".json"):
                    # More robust parsing of library files
                    prover_name = None
                    if filename.startswith("coq_"):
                        prover_name = "coq"
                    elif filename.startswith("lean_"):
                        prover_name = "lean"
                    else:
                        # Try to determine prover from content
                        try:
                            file_path = os.path.join(libraries_dir, filename)
                            with open(file_path, "r") as f:
                                content = json.load(f)
                                # Check for prover-specific patterns in content
                                if any("Require Import" in str(v) for v in content.values()):
                                    prover_name = "coq"
                                elif any("import Mathlib" in str(v) for v in content.values()):
                                    prover_name = "lean"
                        except Exception:
                            logger.warning(f"Could not determine prover for: {filename}")
                    
                    if prover_name:
                        file_path = os.path.join(libraries_dir, filename)
                        try:
                            with open(file_path, "r") as f:
                                lib_data = json.load(f)
                                libraries[prover_name].update(lib_data)
                            logger.debug(f"Loaded {len(lib_data)} libraries for {prover_name} from {filename}")
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"Libraries directory not found: {libraries_dir}")
        
        return libraries
    
    def _load_patterns(self) -> Dict[str, Any]:
        """
        Load proof patterns from JSON files.
        
        Returns:
            Dictionary of pattern information
        """
        patterns = {}
        patterns_dir = os.path.join(self.data_dir, "patterns")
        
        if os.path.exists(patterns_dir):
            for filename in os.listdir(patterns_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(patterns_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            pattern_data = json.load(f)
                            patterns.update(pattern_data)
                        logger.debug(f"Loaded {len(pattern_data)} patterns from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"Patterns directory not found: {patterns_dir}")
        
        return patterns
    
    def _load_msc_categories(self) -> Dict[str, Any]:
        """
        Load MSC (Mathematics Subject Classification) categories.
        
        Returns:
            Dictionary of MSC categories
        """
        msc_path = os.path.join(self.data_dir, "msc_categories.json")
        
        if os.path.exists(msc_path):
            try:
                with open(msc_path, "r") as f:
                    msc_data = json.load(f)
                logger.debug(f"Loaded {len(msc_data)} MSC categories")
                return msc_data
            except Exception as e:
                logger.error(f"Error loading MSC categories: {e}")
        else:
            logger.warning(f"MSC categories file not found: {msc_path}")
        
        return {}
    
    def get_concept_mapping(self, concept: str, domain: Optional[str] = None, prover: str = "coq") -> str:
        """
        Map a mathematical concept to its prover representation.
        
        Args:
            concept: The concept to map
            domain: Optional domain for context
            prover: The target theorem prover
            
        Returns:
            The prover-specific representation
        """
        # Check if concept exists in knowledge base
        if concept in self.concepts:
            concept_info = self.concepts[concept]
            
            # Check for formal definition
            if "formal_definition" in concept_info and prover in concept_info["formal_definition"]:
                return concept_info["formal_definition"][prover]
            
            # Check domain-specific definition
            if domain and "domains" in concept_info and domain in concept_info["domains"]:
                domain_info = concept_info["domains"][domain]
                if "formal_definition" in domain_info and prover in domain_info["formal_definition"]:
                    return domain_info["formal_definition"][prover]
        
        # Default to returning the concept unchanged
        return concept
    
    def get_libraries_for_concept(self, concept: str, domain: str, prover: str) -> List[str]:
        """
        Get libraries needed for a concept in a specific prover.
        
        Args:
            concept: The concept name
            domain: The mathematical domain
            prover: The target theorem prover
            
        Returns:
            List of library imports
        """
        if concept in self.concepts:
            concept_info = self.concepts[concept]
            if "domains" in concept_info and domain in concept_info["domains"]:
                domain_info = concept_info["domains"][domain]
                return domain_info.get("libraries", {}).get(prover, [])
        
        # Check if concept exists directly in libraries
        if prover in self.libraries:
            for lib_name, lib_info in self.libraries[prover].items():
                if "provides" in lib_info and concept in lib_info["provides"]:
                    return [lib_info.get("import", lib_name)]
        
        return []
    
    def get_tactics_for_pattern(self, pattern: str, prover: str) -> List[TacticInfo]:
        """
        Get appropriate tactics for a proof pattern in a specific prover.
        
        Args:
            pattern: The proof pattern name
            prover: The target theorem prover
            
        Returns:
            List of tactic information
        """
        tactics = []
        
        if pattern in self.patterns:
            pattern_info = self.patterns[pattern]
            if "tactics" in pattern_info and prover in pattern_info["tactics"]:
                raw_tactics = pattern_info["tactics"][prover]
                
                # Convert raw tactics to TacticInfo objects
                for tactic_data in raw_tactics:
                    if isinstance(tactic_data, dict):
                        # Extract tactic information
                        tactic_str = tactic_data.get("tactic", "")
                        description = tactic_data.get("description", "")
                        
                        # Parse tactic type and arguments
                        tactic_type, arguments = self._parse_tactic(tactic_str)
                        
                        # Create TacticInfo object
                        tactics.append(TacticInfo(
                            tactic_type=tactic_type,
                            arguments=arguments,
                            description=description
                        ))
        
        return tactics
    
    def _parse_tactic(self, tactic_str: str) -> tuple[TacticType, List[str]]:
        """
        Parse a tactic string into type and arguments.
        
        Args:
            tactic_str: The tactic string (e.g., "induction n")
            
        Returns:
            Tuple of (tactic_type, arguments)
        """
        parts = tactic_str.split(maxsplit=1)
        tactic_name = parts[0]
        arguments = parts[1:] if len(parts) > 1 else []
        
        # Extract variable args from placeholder format
        if arguments and "{var}" in arguments[0]:
            arguments = [arguments[0].replace("{var}", "n")]
        
        # Map tactic name to type
        tactic_type_map = {
            "induction": TacticType.INDUCTION,
            "intros": TacticType.INTRO,
            "intro": TacticType.INTRO,
            "apply": TacticType.APPLY,
            "rewrite": TacticType.REWRITE,
            "rw": TacticType.REWRITE,
            "cases": TacticType.CASE_ANALYSIS,
            "contradiction": TacticType.CONTRADICTION,
            "exists": TacticType.EXISTS,
            "use": TacticType.EXISTS,
            "ring": TacticType.CUSTOM,
            "field": TacticType.CUSTOM,
            "simp": TacticType.SIMPLIFY,
            "simpl": TacticType.SIMPLIFY,
            "auto": TacticType.AUTO,
            "tauto": TacticType.AUTO,
            "lia": TacticType.CUSTOM,
            "linarith": TacticType.CUSTOM,
            "norm_num": TacticType.CUSTOM
        }
        
        tactic_type = tactic_type_map.get(tactic_name, TacticType.CUSTOM)
        
        # For CUSTOM tactics, include the tactic name as the first argument
        if tactic_type == TacticType.CUSTOM:
            return tactic_type, [tactic_name] + arguments
        
        return tactic_type, arguments
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """
        Get information about a mathematical domain.
        
        Args:
            domain: The mathematical domain code
            
        Returns:
            Dictionary with domain information
        """
        # First check if it's a direct match
        if domain in self.msc_categories:
            return self.msc_categories[domain]
        
        # Check for domain ranges (e.g., "12-20")
        if "-" in domain:
            start, end = domain.split("-")
            # Return info for the start of the range if available
            if start in self.msc_categories:
                return self.msc_categories[start]
        
        return {}
    
    def get_domain_libraries(self, domain: str, prover: str) -> List[str]:
        """
        Get recommended libraries for a domain in a specific prover.
        
        Args:
            domain: The mathematical domain code
            prover: The target theorem prover
            
        Returns:
            List of library imports
        """
        libraries = []
        
        # Common domain-specific libraries
        domain_libraries = {
            # Number theory
            "11": {
                "coq": ["Arith", "ZArith", "Lia"],
                "lean": ["Mathlib.Data.Nat.Basic", "Mathlib.Data.Nat.Parity", "Mathlib.Tactic.Ring"]
            },
            # Algebra domains
            "12-20": {
                "coq": ["Ring", "Field", "Algebra"],
                "lean": ["Mathlib.Algebra.Ring.Basic", "Mathlib.Tactic.Ring"]
            },
            # Analysis domains
            "26-42": {
                "coq": ["Reals", "Ranalysis"],
                "lean": ["Mathlib.Analysis.RealFunction"]
            },
            # Topology domains
            "54-55": {
                "coq": ["Topology"],
                "lean": ["Mathlib.Topology.Basic"]
            }
        }
        
        # Check if we have predefined libraries for this domain
        if domain in domain_libraries:
            return domain_libraries[domain].get(prover, [])
        
        # Check for domain ranges
        for domain_range, libs in domain_libraries.items():
            if "-" in domain_range:
                start, end = domain_range.split("-")
                if start <= domain <= end:
                    return libs.get(prover, [])
        
        return libraries
    
    def get_examples(self, domain: str, pattern: str, prover: str) -> List[Dict[str, str]]:
        """
        Get example proofs for a domain and pattern.
        
        Args:
            domain: The mathematical domain
            pattern: The proof pattern
            prover: The target theorem prover
            
        Returns:
            List of example proofs
        """
        examples = []
        
        # Try to find examples in patterns
        if pattern in self.patterns:
            pattern_info = self.patterns[pattern]
            if "examples" in pattern_info and prover in pattern_info["examples"]:
                return pattern_info["examples"][prover]
        
        return examples
    
    def get_notation_mappings(self, domain: str, prover: str) -> Dict[str, str]:
        """
        Get notation mappings for a domain and prover.
        
        Args:
            domain: The mathematical domain
            prover: The target theorem prover
            
        Returns:
            Dictionary mapping notation patterns to replacements
        """
        # Default notation mappings
        default_mappings = {
            "coq": {
                r'\bfor\s+all\s+([a-zA-Z][a-zA-Z0-9]*)\b': r'forall \1,',
                r'\bthere\s+exists\s+([a-zA-Z][a-zA-Z0-9]*)\b': r'exists \1,',
                r'\band\b': r'/\\',
                r'\bor\b': r'\\/',
                r'\bnot\b': r'~'
            },
            "lean": {
                r'\bfor\s+all\s+([a-zA-Z][a-zA-Z0-9]*)\b': r'∀ \1,',
                r'\bthere\s+exists\s+([a-zA-Z][a-zA-Z0-9]*)\b': r'∃ \1,',
                r'\band\b': r'∧',
                r'\bor\b': r'∨',
                r'\bnot\b': r'¬'
            }
        }
        
        # Get default mappings for the prover
        mappings = default_mappings.get(prover, {})
        
        # Try to find domain-specific mappings
        # (This would be expanded with actual domain-specific notations)
        
        return mappings
    
    def get_pattern_config(self, pattern: str, prover: str) -> Dict[str, Any]:
        """
        Get configuration for a proof pattern.
        
        Args:
            pattern: The proof pattern name
            prover: The target theorem prover
            
        Returns:
            Dictionary with pattern configuration
        """
        if pattern in self.patterns:
            pattern_info = self.patterns[pattern]
            
            # Try to get a config specific to the prover
            if "translation_config" in pattern_info:
                if prover in pattern_info["translation_config"]:
                    return pattern_info["translation_config"][prover]
                elif "default" in pattern_info["translation_config"]:
                    return pattern_info["translation_config"]["default"]
                
            # Build a basic config if no specific config exists
            return {
                "parameters": {},
                "suggested_tactics": self.get_tactics_for_pattern(pattern, prover)
            }
        
        return {"parameters": {}, "suggested_tactics": []}

# Utility functions

def create_knowledge_base(data_dir="knowledge_base/data") -> SimpleKnowledgeBase:
    """Create a new knowledge base instance."""
    return SimpleKnowledgeBase(data_dir)