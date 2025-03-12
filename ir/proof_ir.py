"""
Intermediate Representation (IR) for mathematical proofs.
Serves as a system-agnostic representation between natural language and formal proofs.
"""

from enum import Enum
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

class NodeType(Enum):
    """Types of nodes in the proof AST."""
    THEOREM = "theorem"               # Theorem statement
    ASSUMPTION = "assumption"         # Assumption/hypothesis
    STEP = "step"                     # Proof step
    CASE = "case"                     # Case in a case analysis
    INDUCTION_BASE = "induction_base" # Base case in induction
    INDUCTION_STEP = "induction_step" # Inductive step
    CONTRADICTION = "contradiction"   # Contradiction statement
    EXISTS_WITNESS = "exists_witness" # Existential witness
    FORALL_INST = "forall_inst"       # Universal instantiation
    REFERENCE = "reference"           # Reference to previous statement
    CONCLUSION = "conclusion"         # Proof conclusion

class ExprType(Enum):
    """Types of mathematical expressions."""
    VARIABLE = "variable"             # Single variable
    NUMBER = "number"                 # Numeric constant
    OPERATION = "operation"           # Mathematical operation
    FUNCTION = "function"             # Function application
    RELATION = "relation"             # Relation between expressions
    QUANTIFIER = "quantifier"         # Quantified expression
    LOGICAL = "logical"               # Logical connective
    SET = "set"                       # Set expression
    
class TacticType(Enum):
    """Common proof tactics across systems."""
    INTRO = "intro"                   # Introduce variables/hypotheses
    APPLY = "apply"                   # Apply a theorem/lemma
    REWRITE = "rewrite"               # Rewrite with an equation
    CASE_ANALYSIS = "case_analysis"   # Split into cases
    INDUCTION = "induction"           # Proof by induction
    CONTRADICTION = "contradiction"   # Proof by contradiction
    EXISTS = "exists"                 # Provide existential witness
    SIMPLIFY = "simplify"             # Simplify expressions
    AUTO = "auto"                     # Automatic proof search
    CUSTOM = "custom"                 # Custom tactic

@dataclass
class Expression:
    """
    Representation of a mathematical expression in the IR.
    """
    expr_type: ExprType
    value: Any
    children: List['Expression'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProofNode:
    """
    Node in the proof AST representing a single step or structure.
    """
    node_type: NodeType
    content: Union[str, Expression]
    children: List['ProofNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TacticInfo:
    """
    Information about a proof tactic.
    """
    tactic_type: TacticType
    arguments: List[Any] = field(default_factory=list)
    description: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LibraryDependency:
    """
    Represents a dependency on a library or theorem.
    """
    name: str
    import_path: str
    concepts: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.name, self.import_path))
    
    def __eq__(self, other):
        if not isinstance(other, LibraryDependency):
            return False
        return self.name == other.name and self.import_path == other.import_path

@dataclass
class DomainContext:
    """
    Represents domain-specific context for a proof.
    """
    domain: str
    subdomain: str = None
    libraries: List[str] = field(default_factory=list)
    theorems: List[str] = field(default_factory=list)
    axioms: List[str] = field(default_factory=list)
    notations: Dict[str, str] = field(default_factory=dict)

@dataclass
class ProofIR:
    """
    Complete intermediate representation of a mathematical proof.
    """
    # Theorem statement
    theorem: ProofNode
    
    # Main proof tree
    proof_tree: List[ProofNode]
    
    # Domain information
    domain: Dict[str, Any]
    
    # Pattern information
    pattern: Dict[str, Any]
    
    # Suggested tactics
    tactics: List[TacticInfo] = field(default_factory=list)
    
    # Original text
    original_theorem: str = None
    original_proof: str = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Domain-specific context
    domain_context: DomainContext = None
    
    # Library dependencies
    library_dependencies: List[LibraryDependency] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize domain context if not provided."""
        if self.domain_context is None:
            self.domain_context = DomainContext(self.domain.get("primary_domain", ""))
    
    def add_library_dependency(self, name: str, import_path: str, concepts: List[str] = None):
        """
        Add a library dependency to the IR.
        
        Args:
            name: The library name
            import_path: The import path
            concepts: Optional list of concepts requiring this library
        """
        concepts = concepts or []
        dependency = LibraryDependency(name, import_path, concepts)
        if dependency not in self.library_dependencies:
            self.library_dependencies.append(dependency)
    
    def to_dict(self) -> Dict:
        """Convert the IR to a dictionary for serialization."""
        return {
            "theorem": self._node_to_dict(self.theorem),
            "proof_tree": [self._node_to_dict(node) for node in self.proof_tree],
            "domain": self.domain,
            "pattern": self.pattern,
            "tactics": [
                {
                    "tactic_type": tactic.tactic_type.value,
                    "arguments": tactic.arguments,
                    "description": tactic.description,
                    "metadata": tactic.metadata
                }
                for tactic in self.tactics
            ],
            "original_theorem": self.original_theorem,
            "original_proof": self.original_proof,
            "metadata": self.metadata,
            "domain_context": {
                "domain": self.domain_context.domain,
                "subdomain": self.domain_context.subdomain,
                "libraries": self.domain_context.libraries,
                "theorems": self.domain_context.theorems,
                "axioms": self.domain_context.axioms,
                "notations": self.domain_context.notations
            },
            "library_dependencies": [
                {
                    "name": dep.name,
                    "import_path": dep.import_path,
                    "concepts": dep.concepts
                }
                for dep in self.library_dependencies
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProofIR':
        """Create an IR instance from a dictionary."""
        domain_context = DomainContext(
            domain=data["domain_context"]["domain"],
            subdomain=data["domain_context"]["subdomain"],
            libraries=data["domain_context"]["libraries"],
            theorems=data["domain_context"]["theorems"],
            axioms=data["domain_context"]["axioms"],
            notations=data["domain_context"]["notations"]
        )
        
        library_dependencies = [
            LibraryDependency(
                name=dep["name"],
                import_path=dep["import_path"],
                concepts=dep["concepts"]
            )
            for dep in data["library_dependencies"]
        ]
        
        ir = cls(
            theorem=cls._dict_to_node(data["theorem"]),
            proof_tree=[cls._dict_to_node(node) for node in data["proof_tree"]],
            domain=data["domain"],
            pattern=data["pattern"],
            tactics=[
                TacticInfo(
                    tactic_type=TacticType(t["tactic_type"]),
                    arguments=t["arguments"],
                    description=t["description"],
                    metadata=t["metadata"]
                )
                for t in data["tactics"]
            ],
            original_theorem=data["original_theorem"],
            original_proof=data["original_proof"],
            metadata=data["metadata"],
            domain_context=domain_context,
            library_dependencies=library_dependencies
        )
        
        return ir
    
    @staticmethod
    def _node_to_dict(node: ProofNode) -> Dict:
        """Convert a node to a dictionary."""
        result = {
            "node_type": node.node_type.value,
            "content": node.content if isinstance(node.content, str) else {
                "expr_type": node.content.expr_type.value,
                "value": node.content.value,
                "children": [
                    {
                        "expr_type": child.expr_type.value,
                        "value": child.value,
                        "children": [],  # Simplified for brevity
                        "metadata": child.metadata
                    }
                    for child in (node.content.children or [])
                ],
                "metadata": node.content.metadata
            },
            "children": [],
            "metadata": node.metadata
        }
        
        if node.children:
            result["children"] = [ProofIR._node_to_dict(child) for child in node.children]
        
        return result
    
    @staticmethod
    def _dict_to_node(data: Dict) -> ProofNode:
        """Convert a dictionary to a node."""
        content = data["content"]
        if isinstance(content, dict):
            content = Expression(
                expr_type=ExprType(content["expr_type"]),
                value=content["value"],
                children=[
                    Expression(
                        expr_type=ExprType(child["expr_type"]),
                        value=child["value"],
                        children=[],  # Simplified for brevity
                        metadata=child["metadata"]
                    )
                    for child in content["children"]
                ],
                metadata=content["metadata"]
            )
        
        node = ProofNode(
            node_type=NodeType(data["node_type"]),
            content=content,
            metadata=data["metadata"]
        )
        
        if data["children"]:
            node.children = [ProofIR._dict_to_node(child) for child in data["children"]]
        
        return node

# Helper functions to create common IR structures
def create_theorem_node(statement: str) -> ProofNode:
    """Create a simple theorem node from a string statement."""
    return ProofNode(
        node_type=NodeType.THEOREM,
        content=statement
    )

def create_assumption_node(assumption: str) -> ProofNode:
    """Create an assumption node from a string."""
    return ProofNode(
        node_type=NodeType.ASSUMPTION,
        content=assumption
    )

def create_step_node(step: str) -> ProofNode:
    """Create a proof step node from a string."""
    return ProofNode(
        node_type=NodeType.STEP,
        content=step
    )

def create_conclusion_node(conclusion: str) -> ProofNode:
    """Create a conclusion node from a string."""
    return ProofNode(
        node_type=NodeType.CONCLUSION,
        content=conclusion
    )

def create_simple_proof_ir(theorem: str, proof: str, domain_info: Dict[str, Any] = None) -> ProofIR:
    """
    Create a simple proof IR from theorem statement and proof text.
    For quick prototyping without full parsing.
    """
    # Create a basic proof tree with just theorem, assumption, and conclusion
    theorem_node = create_theorem_node(theorem)
    
    # Split proof into sentences
    steps = [s.strip() for s in proof.split('.') if s.strip()]
    
    proof_tree = []
    if steps:
        # First sentence is usually an assumption
        proof_tree.append(create_assumption_node(steps[0]))
        
        # Middle sentences are proof steps
        for step in steps[1:-1]:
            if step:
                proof_tree.append(create_step_node(step))
        
        # Last sentence is usually the conclusion
        if len(steps) > 1:
            proof_tree.append(create_conclusion_node(steps[-1]))
    
    # Default domain and pattern info
    if domain_info is None:
        domain_info = {
            "msc_category": "00",
            "msc_name": "General Mathematics",
            "proof_technique": "direct proof"
        }
    
    pattern_info = {
        "name": "unknown",
        "confidence": 0.0
    }
    
    # Create the IR
    return ProofIR(
        theorem=theorem_node,
        proof_tree=proof_tree,
        domain=domain_info,
        pattern=pattern_info,
        original_theorem=theorem,
        original_proof=proof
    )