"""
Library of Coq tactics with domain-specific knowledge.
"""

from typing import List, Dict, Any, Optional

class TacticsLibrary:
    """
    Library of Coq tactics for different proof patterns and domains.
    """
    
    def __init__(self):
        """Initialize tactics library."""
        # Core tactics used in most proofs
        self.core_tactics = {
            "intro": "intro {var}",
            "intros": "intros",
            "simpl": "simpl",
            "auto": "auto"
        }
        
        # Domain-specific tactics
        self.domain_tactics = {
            "11": {  # Number theory
                "lia": "lia",
                "ring": "ring",
                "omega": "omega"
            },
            "12-20": {  # Algebra
                "ring": "ring",
                "field": "field"
            },
            "26-42": {  # Analysis
                "field": "field"
            }
        }
        
        # Pattern-specific tactics
        self.pattern_tactics = {
            "evenness": [
                ("intro", ["{var}"]),
                ("exists", ["{var}"]),
                ("ring", [])
            ],
            "induction": [
                ("induction", ["{var}"]),
                ("simpl", []),
                ("auto", [])
            ],
            "contradiction": [
                ("intros", []),
                ("intro", ["H"]),
                ("contradiction", [])
            ],
            "cases": [
                ("intros", []),
                ("destruct", ["{var}"]),
                ("simpl", []),
                ("auto", [])
            ]
        }
    
    def get_tactics_for_pattern(self, pattern: str, variables: List[str]) -> List[str]:
        """Get tactics for a specific proof pattern."""
        tactics = []
        
        if pattern in self.pattern_tactics:
            for tactic_name, args in self.pattern_tactics[pattern]:
                # Format tactic with variables
                formatted_args = []
                for arg in args:
                    if arg == "{var}" and variables:
                        formatted_args.append(variables[0])
                    else:
                        formatted_args.append(arg)
                
                # Build tactic string
                if formatted_args:
                    tactics.append(f"{tactic_name} {' '.join(formatted_args)}.")
                else:
                    tactics.append(f"{tactic_name}.")
        
        return tactics
    
    def get_tactics_for_domain(self, domain: str) -> List[str]:
        """Get domain-specific tactics."""
        tactics = []
        
        if domain in self.domain_tactics:
            for tactic_name in self.domain_tactics[domain]:
                tactics.append(f"{tactic_name}.")
        
        return tactics