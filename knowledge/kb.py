"""
Simplified knowledge base for the proof translator system.
"""

import json
import os
from typing import Dict, List, Any, Optional

class KnowledgeBase:
    """
    Lightweight knowledge base for mathematical domains and proof patterns.
    """
    
    def __init__(self, data_dir="knowledge/data"):
        """
        Initialize the knowledge base.
        
        Args:
            data_dir: Directory containing knowledge data files
        """
        self.data_dir = data_dir
        
        # Load knowledge data
        self.domains = self._load_json("domains.json")
        self.patterns = self._load_json("patterns.json")
        self.tactics = self._load_json("tactics.json")
        
        print(f"Loaded knowledge base with {len(self.domains)} domains and {len(self.patterns)} patterns")
    
    def _load_json(self, filename: str) -> Dict:
        """Load data from a JSON file."""
        path = os.path.join(self.data_dir, filename)
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty dict if file not found or invalid
            return {}
    
    def get_domain_info(self, domain_code: str) -> Dict[str, Any]:
        """Get information about a mathematical domain."""
        if domain_code in self.domains:
            return self.domains[domain_code]
        return {}
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get information about a proof pattern."""
        if pattern_name in self.patterns:
            return self.patterns[pattern_name]
        return {}
    
    def get_domain_tactics(self, domain_code: str) -> List[Dict[str, str]]:
        """Get recommended tactics for a domain."""
        if domain_code in self.tactics.get("domains", {}):
            return self.tactics["domains"][domain_code]
        return []
    
    def get_pattern_tactics(self, pattern_name: str) -> List[Dict[str, str]]:
        """Get recommended tactics for a pattern."""
        if pattern_name in self.tactics.get("patterns", {}):
            return self.tactics["patterns"][pattern_name]
        return []
    
    def get_imports_for_domain(self, domain_code: str) -> List[str]:
        """Get required Coq imports for a domain."""
        imports = []
        
        # Core imports that are always needed
        imports.append("Require Import Arith.")
        
        # Domain-specific imports
        if domain_code == "11":  # Number theory
            imports.extend([
                "Require Import Lia.",
                "Require Import ZArith."
            ])
        elif domain_code in ["12", "13", "14", "15", "16", "17", "18", "19", "20"]:  # Algebra
            imports.extend([
                "Require Import Ring.",
                "Require Import Field."
            ])
        elif domain_code in ["26", "27", "28", "30", "31", "32", "33", "34", "35"]:  # Analysis
            imports.extend([
                "Require Import Reals."
            ])
            
        # Get additional imports from the knowledge base
        domain_info = self.get_domain_info(domain_code)
        if "imports" in domain_info:
            imports.extend(domain_info["imports"])
        
        # Deduplicate
        return list(dict.fromkeys(imports))
    
    def get_tactic_string(self, tactic_name: str, args: Optional[List[str]] = None) -> str:
        """Get the Coq syntax for a tactic."""
        args_str = " ".join(args) if args else ""
        
        # Map tactic names to Coq syntax
        tactic_map = {
            "intro": f"intro {args_str}",
            "intros": f"intros {args_str}",
            "apply": f"apply {args_str}",
            "rewrite": f"rewrite {args_str}",
            "destruct": f"destruct {args_str}",
            "induction": f"induction {args_str}",
            "exists": f"exists {args_str}",
            "ring": "ring",
            "lia": "lia",
            "auto": "auto",
            "simpl": "simpl",
            "reflexivity": "reflexivity",
            "contradiction": "contradiction"
        }
        
        return tactic_map.get(tactic_name, f"{tactic_name} {args_str}")