# knowledge_base/simple_kb.py
import json
import os
from typing import Dict, List, Any, Optional

class SimpleKnowledgeBase:
    """A simplified knowledge base that loads JSON data files directly."""
    
    def __init__(self, data_dir="knowledge_base/data"):
        self.data_dir = data_dir
        self.concepts = self._load_concepts()
        self.libraries = self._load_libraries()
        self.patterns = self._load_patterns()
        self.msc_categories = self._load_msc_categories()
        
    def _load_concepts(self) -> Dict[str, Any]:
        """Load concept definitions from JSON files."""
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
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return concepts
    
    def _load_libraries(self) -> Dict[str, Any]:
        """Load library information from JSON files."""
        libraries = {"coq": {}, "lean": {}}
        libraries_dir = os.path.join(self.data_dir, "libraries")
        
        if os.path.exists(libraries_dir):
            for filename in os.listdir(libraries_dir):
                if filename.endswith(".json"):
                    prover_name = filename.split("_")[0]
                    file_path = os.path.join(libraries_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            libraries[prover_name] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return libraries
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load proof patterns from JSON files."""
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
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return patterns
    
    def _load_msc_categories(self) -> Dict[str, Any]:
        """Load MSC (Mathematics Subject Classification) categories."""
        msc_path = os.path.join(self.data_dir, "msc_categories.json")
        
        if os.path.exists(msc_path):
            try:
                with open(msc_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading MSC categories: {e}")
        
        return {}
    
    def get_libraries_for_concept(self, concept: str, domain: str, prover: str) -> List[str]:
        """Get libraries needed for a concept in a specific prover."""
        if concept in self.concepts:
            concept_info = self.concepts[concept]
            if "domains" in concept_info and domain in concept_info["domains"]:
                domain_info = concept_info["domains"][domain]
                return domain_info.get("libraries", {}).get(prover, [])
        return []
    
    def get_tactics_for_pattern(self, pattern: str, prover: str) -> List[Any]:
        """Get appropriate tactics for a proof pattern in a specific prover."""
        if pattern in self.patterns:
            pattern_info = self.patterns[pattern]
            return pattern_info.get("tactics", {}).get(prover, [])
        return []
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """Get information about a mathematical domain."""
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
        """Get recommended libraries for a domain in a specific prover."""
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