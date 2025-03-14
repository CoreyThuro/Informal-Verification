"""
Domain knowledge base for mathematical domains and proof translation.
Provides centralized knowledge about mathematical concepts, libraries, and proof patterns.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Set, Union

# Configure logging
logger = logging.getLogger("knowledge_base")

class DomainKnowledgeBase:
    """
    Centralized repository for domain-specific mathematical knowledge.
    
    This class manages knowledge about mathematical domains, concepts,
    libraries, and proof patterns. It provides mappings between concepts
    and library imports, as well as information about theorem provers.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the domain knowledge base.
        
        Args:
            data_dir: Optional directory for knowledge data files.
                     Defaults to 'knowledge_base/data' in the package directory.
        """
        # Set up data directory
        if data_dir is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(package_dir, 'data')
        else:
            self.data_dir = data_dir
            
        # Initialize data structures
        self.domains = {}  # MSC domains
        self.concepts = {}  # Mathematical concepts
        self.libraries = {  # Library information for provers
            "coq": {},
            "lean": {}
        }
        self.patterns = {}  # Proof patterns
        self.tactics = {}   # Domain-specific tactics
        
        # Load knowledge data
        self.load_knowledge()
        
        logger.info(f"Domain knowledge base initialized with {len(self.domains)} domains, "
                   f"{len(self.concepts)} concepts, and {len(self.patterns)} patterns")
    
    def load_knowledge(self):
        """Load domain knowledge from data files."""
        self._load_domains()
        self._load_concepts()
        self._load_libraries()
        self._load_patterns()

    def _load_domains(self):
        """Load MSC domain information."""
        domains_file = os.path.join(self.data_dir, 'msc_categories.json')
        try:
            with open(domains_file, 'r') as f:
                self.domains = json.load(f)
            logger.debug(f"Loaded {len(self.domains)} MSC domains from {domains_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load domains from {domains_file}: {e}")
            # Create basic domain structure
            self._create_default_domains()
    
    def _load_concepts(self):
        """Load mathematical concept information."""
        concepts_dir = os.path.join(self.data_dir, 'concepts')
        if not os.path.exists(concepts_dir):
            os.makedirs(concepts_dir, exist_ok=True)
            logger.warning(f"Created concepts directory: {concepts_dir}")
            self._create_default_concepts()
            return
            
        # Load each concept file
        for filename in os.listdir(concepts_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(concepts_dir, filename), 'r') as f:
                        concepts = json.load(f)
                        self.concepts.update(concepts)
                    logger.debug(f"Loaded concepts from {filename}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not load concepts from {filename}: {e}")
    
    def _load_libraries(self):
        """Load library mappings for different provers."""
        libraries_dir = os.path.join(self.data_dir, 'libraries')
        if not os.path.exists(libraries_dir):
            os.makedirs(libraries_dir, exist_ok=True)
            logger.warning(f"Created libraries directory: {libraries_dir}")
            self._create_default_libraries()
            return
            
        # Load Coq libraries
        coq_file = os.path.join(libraries_dir, 'coq_libraries.json')
        try:
            with open(coq_file, 'r') as f:
                self.libraries["coq"] = json.load(f)
            logger.debug(f"Loaded Coq libraries from {coq_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load Coq libraries from {coq_file}: {e}")
            self._create_default_coq_libraries()
        
        # Load Lean libraries
        lean_file = os.path.join(libraries_dir, 'lean_libraries.json')
        try:
            with open(lean_file, 'r') as f:
                self.libraries["lean"] = json.load(f)
            logger.debug(f"Loaded Lean libraries from {lean_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load Lean libraries from {lean_file}: {e}")
            self._create_default_lean_libraries()
    
    def _load_patterns(self):
        """Load proof pattern information."""
        patterns_dir = os.path.join(self.data_dir, 'patterns')
        if not os.path.exists(patterns_dir):
            os.makedirs(patterns_dir, exist_ok=True)
            logger.warning(f"Created patterns directory: {patterns_dir}")
            self._create_default_patterns()
            return
            
        # Load each pattern file
        for filename in os.listdir(patterns_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(patterns_dir, filename), 'r') as f:
                        pattern_type = filename.replace('.json', '')
                        self.patterns[pattern_type] = json.load(f)
                    logger.debug(f"Loaded patterns from {filename}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not load patterns from {filename}: {e}")
    
    def _load_tactics(self):
        """Load domain-specific tactics information."""
        tactics_dir = os.path.join(self.data_dir, 'tactics')
        if not os.path.exists(tactics_dir):
            os.makedirs(tactics_dir, exist_ok=True)
            logger.warning(f"Created tactics directory: {tactics_dir}")
            self._create_default_tactics()
            return
            
        # Load each tactics file
        for filename in os.listdir(tactics_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(tactics_dir, filename), 'r') as f:
                        domain = filename.replace('.json', '')
                        self.tactics[domain] = json.load(f)
                    logger.debug(f"Loaded tactics from {filename}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not load tactics from {filename}: {e}")
    
    def _create_default_domains(self):
        """Create default MSC domain information."""
        self.domains = {
            "00": {"name": "General Mathematics", "description": "General mathematical topics"},
            "11": {"name": "Number Theory", "description": "Number theory, including divisibility, congruences, and primality"},
            "12": {"name": "Field Theory and Polynomials", "description": "Field theory and polynomials"},
            "13": {"name": "Commutative Algebra", "description": "Commutative algebra, rings, and ideals"},
            "14": {"name": "Algebraic Geometry", "description": "Algebraic geometry"},
            "15": {"name": "Linear Algebra", "description": "Linear and multilinear algebra, matrix theory"},
            "20": {"name": "Group Theory", "description": "Group theory and generalizations"},
            "26": {"name": "Real Functions", "description": "Real functions, analysis"},
            "30": {"name": "Functions of a Complex Variable", "description": "Complex analysis"},
            "54": {"name": "General Topology", "description": "General topology"},
            "55": {"name": "Algebraic Topology", "description": "Algebraic topology"}
        }
        
        # Save to file
        domains_file = os.path.join(self.data_dir, 'msc_categories.json')
        os.makedirs(os.path.dirname(domains_file), exist_ok=True)
        with open(domains_file, 'w') as f:
            json.dump(self.domains, f, indent=2)
        logger.info(f"Created default MSC domains in {domains_file}")
    
    def _create_default_concepts(self):
        """Create default mathematical concept information."""
        # Number Theory concepts
        number_theory_concepts = {
            "even": {
                "definition": "An integer divisible by 2",
                "formal_definition": {
                    "coq": "exists k, n = 2 * k",
                    "lean": "∃ k, n = 2 * k"
                },
                "domains": {
                    "11": {
                        "libraries": {
                            "coq": ["Arith"],
                            "lean": ["Mathlib.Data.Nat.Parity"]
                        }
                    }
                },
                "related_concepts": ["odd", "divisible"]
            },
            "odd": {
                "definition": "An integer not divisible by 2",
                "formal_definition": {
                    "coq": "exists k, n = 2 * k + 1",
                    "lean": "∃ k, n = 2 * k + 1"
                },
                "domains": {
                    "11": {
                        "libraries": {
                            "coq": ["Arith"],
                            "lean": ["Mathlib.Data.Nat.Parity"]
                        }
                    }
                },
                "related_concepts": ["even", "divisible"]
            },
            "prime": {
                "definition": "A natural number greater than 1 that is not a product of two smaller natural numbers",
                "formal_definition": {
                    "coq": "prime n",
                    "lean": "Nat.Prime n"
                },
                "domains": {
                    "11": {
                        "libraries": {
                            "coq": ["Znumtheory"],
                            "lean": ["Mathlib.NumberTheory.Primes"]
                        }
                    }
                },
                "related_concepts": ["composite", "divisible"]
            }
        }
        
        # Save number theory concepts
        nt_file = os.path.join(self.data_dir, 'concepts', 'number_theory.json')
        os.makedirs(os.path.dirname(nt_file), exist_ok=True)
        with open(nt_file, 'w') as f:
            json.dump(number_theory_concepts, f, indent=2)
        
        # Update concepts dictionary
        self.concepts.update(number_theory_concepts)
        logger.info(f"Created default number theory concepts in {nt_file}")
    
    def _create_default_coq_libraries(self):
        """Create default Coq library information."""
        coq_libraries = {
            "Arith": {
                "import": "Require Import Arith.",
                "description": "Basic arithmetic library",
                "provides": ["nat", "plus", "mult", "minus", "le", "lt", "max", "min", "even", "odd"],
                "domains": ["11"]
            },
            "ZArith": {
                "import": "Require Import ZArith.",
                "description": "Integer arithmetic library",
                "provides": ["Z", "Z.add", "Z.mul", "Z.sub", "Z.div", "Z.modulo", "Z.abs", "Z.even", "Z.odd", "Z.gcd"],
                "domains": ["11"]
            },
            "Lia": {
                "import": "Require Import Lia.",
                "description": "Linear integer arithmetic solver",
                "provides": ["lia"],
                "domains": ["11"]
            },
            "Ring": {
                "import": "Require Import Ring.",
                "description": "Ring algebraic structure and tactics",
                "provides": ["ring", "ring_simplify"],
                "domains": ["12-20"]
            }
        }
        
        # Save Coq libraries
        coq_file = os.path.join(self.data_dir, 'libraries', 'coq_libraries.json')
        os.makedirs(os.path.dirname(coq_file), exist_ok=True)
        with open(coq_file, 'w') as f:
            json.dump(coq_libraries, f, indent=2)
        
        # Update libraries dictionary
        self.libraries["coq"] = coq_libraries
        logger.info(f"Created default Coq libraries in {coq_file}")
    
    def _create_default_lean_libraries(self):
        """Create default Lean library information."""
        lean_libraries = {
            "Mathlib.Data.Nat.Basic": {
                "import": "import Mathlib.Data.Nat.Basic",
                "description": "Basic natural number library",
                "provides": ["Nat", "Nat.add", "Nat.mul", "Nat.sub", "Nat.le", "Nat.lt", "Nat.max", "Nat.min"],
                "domains": ["11"]
            },
            "Mathlib.Data.Nat.Parity": {
                "import": "import Mathlib.Data.Nat.Parity",
                "description": "Even and odd natural numbers",
                "provides": ["Even", "Odd"],
                "domains": ["11"]
            },
            "Mathlib.Tactic.Ring": {
                "import": "import Mathlib.Tactic.Ring",
                "description": "Ring tactic for algebraic simplification",
                "provides": ["ring"],
                "domains": ["11", "12-20"]
            }
        }
        
        # Save Lean libraries
        lean_file = os.path.join(self.data_dir, 'libraries', 'lean_libraries.json')
        os.makedirs(os.path.dirname(lean_file), exist_ok=True)
        with open(lean_file, 'w') as f:
            json.dump(lean_libraries, f, indent=2)
        
        # Update libraries dictionary
        self.libraries["lean"] = lean_libraries
        logger.info(f"Created default Lean libraries in {lean_file}")
    
    def _create_default_patterns(self):
        """Create default proof pattern information."""
        induction_patterns = {
            "induction": {
                "description": "Proof by induction",
                "structure": ["base_case", "inductive_step", "conclusion"],
                "keywords": ["induction", "base case", "inductive", "hypothesis", "step", "k", "k+1"],
                "tactics": {
                    "coq": [
                        {"tactic": "induction {var}", "description": "Apply induction on variable {var}"},
                        {"tactic": "simpl", "description": "Simplify the goal"}
                    ],
                    "lean": [
                        {"tactic": "induction {var}", "description": "Apply induction on variable {var}"},
                        {"tactic": "simp", "description": "Simplify the goal"}
                    ]
                },
                "examples": {
                    "coq": [
                        {
                            "theorem": "forall n : nat, n + 0 = n",
                            "proof": "intros n. induction n. simpl. reflexivity. simpl. rewrite IHn. reflexivity."
                        }
                    ],
                    "lean": [
                        {
                            "theorem": "∀ n : Nat, n + 0 = n",
                            "proof": "by\n  intro n\n  induction n\n  case zero => rfl\n  case succ => simp [*]"
                        }
                    ]
                }
            }
        }
        
        # Save induction patterns
        induction_file = os.path.join(self.data_dir, 'patterns', 'induction_patterns.json')
        os.makedirs(os.path.dirname(induction_file), exist_ok=True)
        with open(induction_file, 'w') as f:
            json.dump(induction_patterns, f, indent=2)
        
        # Update patterns dictionary
        self.patterns.update(induction_patterns)
        logger.info(f"Created default induction patterns in {induction_file}")
    
    def _create_default_tactics(self):
        """Create default domain-specific tactics."""
        number_theory_tactics = {
            "general": [
                {"name": "lia", "prover": "coq", "description": "Linear integer arithmetic solver"},
                {"name": "linarith", "prover": "lean", "description": "Linear arithmetic solver"}
            ],
            "induction": [
                {"name": "induction {var}", "prover": "coq", "description": "Induction on variable"},
                {"name": "induction {var}", "prover": "lean", "description": "Induction on variable"}
            ],
            "evenness": [
                {"name": "exists {var}", "prover": "coq", "description": "Provide witness for evenness"},
                {"name": "use {var}", "prover": "lean", "description": "Provide witness for evenness"},
                {"name": "ring", "prover": "coq", "description": "Solve with ring arithmetic"},
                {"name": "ring", "prover": "lean", "description": "Solve with ring arithmetic"}
            ]
        }
        
        # Save number theory tactics
        nt_tactics_file = os.path.join(self.data_dir, 'tactics', '11.json')
        os.makedirs(os.path.dirname(nt_tactics_file), exist_ok=True)
        with open(nt_tactics_file, 'w') as f:
            json.dump(number_theory_tactics, f, indent=2)
        
        # Update tactics dictionary
        self.tactics["11"] = number_theory_tactics
        logger.info(f"Created default number theory tactics in {nt_tactics_file}")
    
    def get_domain_info(self, domain_code: str) -> Dict[str, Any]:
        """
        Get information about a mathematical domain.
        
        Args:
            domain_code: The MSC code for the domain
            
        Returns:
            Dictionary with domain information
        """
        return self.domains.get(domain_code, {"name": "Unknown", "description": "Unknown domain"})
    
    def get_concept_info(self, concept: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a mathematical concept.
        
        Args:
            concept: The concept name
            domain: Optional domain for context
            
        Returns:
            Dictionary with concept information, or None if not found
        """
        if concept not in self.concepts:
            return None
        
        concept_info = self.concepts[concept]
        if domain and domain in concept_info.get("domains", {}):
            # Merge domain-specific info with general info
            info = concept_info.copy()
            info.update(concept_info["domains"][domain])
            return info
        
        return concept_info
    
    def get_concept_mapping(self, concept: str, domain: Optional[str] = None, prover: str = "coq") -> str:
        """
        Get the mapping of a concept to its representation in a theorem prover.
        
        Args:
            concept: The concept name
            domain: Optional domain for context
            prover: The target theorem prover
            
        Returns:
            The concept representation in the prover's syntax
        """
        concept_info = self.get_concept_info(concept, domain)
        if not concept_info:
            return concept
        
        # Check for formal definition
        if "formal_definition" in concept_info and prover in concept_info["formal_definition"]:
            return concept_info["formal_definition"][prover]
        
        return concept
    
    def get_libraries_for_concept(self, concept: str, domain: Optional[str] = None, prover: str = "coq") -> List[str]:
        """
        Get required library imports for a concept.
        
        Args:
            concept: The concept name
            domain: Optional domain for context
            prover: The target theorem prover
            
        Returns:
            List of library imports
        """
        concept_info = self.get_concept_info(concept, domain)
        if not concept_info:
            return []
        
        # Check for domain-specific libraries
        if domain and "domains" in concept_info and domain in concept_info["domains"]:
            domain_info = concept_info["domains"][domain]
            if "libraries" in domain_info and prover in domain_info["libraries"]:
                return domain_info["libraries"][prover]
        
        # Check for general libraries
        if "libraries" in concept_info and prover in concept_info["libraries"]:
            return concept_info["libraries"][prover]
        
        return []
    
    def get_domain_libraries(self, domain: str, prover: str = "coq") -> List[str]:
        """
        Get library imports for a mathematical domain.
        
        Args:
            domain: The MSC code for the domain
            prover: The target theorem prover
            
        Returns:
            List of library imports for the domain
        """
        libraries = []
        
        # Get libraries for the domain
        for lib_name, lib_info in self.libraries[prover].items():
            if domain in lib_info.get("domains", []):
                libraries.append(lib_info["import"])
        
        return libraries
    
    def get_domain_tactics(self, domain: str, pattern: Optional[str] = None, prover: str = "coq") -> List[Dict[str, str]]:
        """
        Get tactics for a domain and pattern.
        
        Args:
            domain: The MSC code for the domain
            pattern: Optional proof pattern
            prover: The target theorem prover
            
        Returns:
            List of tactics for the domain and pattern
        """
        tactics = []
        
        # Get general tactics for the domain
        if domain in self.tactics:
            for tactic in self.tactics[domain].get("general", []):
                if tactic["prover"] == prover:
                    tactics.append(tactic)
            
            # Add pattern-specific tactics
            if pattern and pattern in self.tactics[domain]:
                for tactic in self.tactics[domain][pattern]:
                    if tactic["prover"] == prover:
                        tactics.append(tactic)
        
        return tactics
    
    def get_pattern_info(self, pattern: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a proof pattern.
        
        Args:
            pattern: The pattern name
            
        Returns:
            Dictionary with pattern information, or None if not found
        """
        for pattern_type, patterns in self.patterns.items():
            if pattern in patterns:
                return patterns[pattern]
        return None
    
    def add_concept(self, concept: str, definition: str, formal_definitions: Dict[str, str], 
                   domains: Dict[str, Dict[str, Any]]) -> None:
        """
        Add a new concept to the knowledge base.
        
        Args:
            concept: The concept name
            definition: The concept definition
            formal_definitions: Dictionary mapping provers to formal definitions
            domains: Dictionary mapping domains to domain-specific information
        """
        self.concepts[concept] = {
            "definition": definition,
            "formal_definition": formal_definitions,
            "domains": domains
        }
        
        # Determine which file to save to based on domains
        domain_codes = list(domains.keys())
        if "11" in domain_codes:
            file_path = os.path.join(self.data_dir, 'concepts', 'number_theory.json')
        elif any(code in ["12", "13", "14", "15", "16", "17", "18", "19", "20"] for code in domain_codes):
            file_path = os.path.join(self.data_dir, 'concepts', 'algebra.json')
        else:
            file_path = os.path.join(self.data_dir, 'concepts', 'general.json')
        
        # Load existing file if it exists
        existing_concepts = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_concepts = json.load(f)
        
        # Add the new concept
        existing_concepts[concept] = self.concepts[concept]
        
        # Save the updated file
        with open(file_path, 'w') as f:
            json.dump(existing_concepts, f, indent=2)
        
        logger.info(f"Added concept '{concept}' to {file_path}")
    
    def update_from_libraries(self) -> None:
        """Update knowledge base from library connectors."""
        # This would be implemented when library connectors are available
        pass


# Create a global instance
knowledge_base = DomainKnowledgeBase()

def get_knowledge_base() -> DomainKnowledgeBase:
    """
    Get the global knowledge base instance.
    
    Returns:
        The global knowledge base instance
    """
    return knowledge_base

def get_domain_info(domain_code: str) -> Dict[str, Any]:
    """
    Get information about a mathematical domain.
    
    Args:
        domain_code: The MSC code for the domain
        
    Returns:
        Dictionary with domain information
    """
    return knowledge_base.get_domain_info(domain_code)

def get_concept_mapping(concept: str, domain: Optional[str] = None, prover: str = "coq") -> str:
    """
    Get the mapping of a concept to its representation in a theorem prover.
    
    Args:
        concept: The concept name
        domain: Optional domain for context
        prover: The target theorem prover
        
    Returns:
        The concept representation in the prover's syntax
    """
    return knowledge_base.get_concept_mapping(concept, domain, prover)

def get_libraries_for_concept(concept: str, domain: Optional[str] = None, prover: str = "coq") -> List[str]:
    """
    Get required library imports for a concept.
    
    Args:
        concept: The concept name
        domain: Optional domain for context
        prover: The target theorem prover
        
    Returns:
        List of library imports
    """
    return knowledge_base.get_libraries_for_concept(concept, domain, prover)