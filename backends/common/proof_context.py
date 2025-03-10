"""
Proof context manager for theorem provers.
Manages the state and context during proof translation and verification.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import re

class ProofContext:
    """
    Manages proof context information during translation and verification.
    Keeps track of available definitions, theorems, and assumptions.
    """
    
    def __init__(self):
        """Initialize the proof context."""
        # Defined variables and their types
        self.variables: Dict[str, str] = {}
        
        # Available assumptions and hypotheses
        self.hypotheses: Dict[str, str] = {}
        
        # Known definitions
        self.definitions: Dict[str, str] = {}
        
        # Available theorems and lemmas
        self.theorems: Dict[str, str] = {}
        
        # Imported libraries
        self.imports: Set[str] = set()
        
        # Current goals
        self.goals: List[str] = []
        
        # Proof metadata
        self.metadata: Dict[str, Any] = {}
    
    def add_variable(self, name: str, type_name: str) -> None:
        """
        Add a variable to the context.
        
        Args:
            name: The variable name
            type_name: The type of the variable
        """
        self.variables[name] = type_name
    
    def add_hypothesis(self, name: str, statement: str) -> None:
        """
        Add a hypothesis to the context.
        
        Args:
            name: The hypothesis name
            statement: The hypothesis statement
        """
        self.hypotheses[name] = statement
    
    def add_definition(self, name: str, definition: str) -> None:
        """
        Add a definition to the context.
        
        Args:
            name: The definition name
            definition: The definition
        """
        self.definitions[name] = definition
    
    def add_theorem(self, name: str, statement: str) -> None:
        """
        Add a theorem to the context.
        
        Args:
            name: The theorem name
            statement: The theorem statement
        """
        self.theorems[name] = statement
    
    def add_import(self, import_name: str) -> None:
        """
        Add an imported library to the context.
        
        Args:
            import_name: The name of the imported library
        """
        self.imports.add(import_name)
    
    def set_goal(self, goal: str) -> None:
        """
        Set the current goal.
        
        Args:
            goal: The goal statement
        """
        self.goals = [goal]
    
    def add_subgoal(self, subgoal: str) -> None:
        """
        Add a subgoal to the list of goals.
        
        Args:
            subgoal: The subgoal statement
        """
        self.goals.append(subgoal)
    
    def remove_goal(self, index: int = 0) -> str:
        """
        Remove a goal from the list.
        
        Args:
            index: The index of the goal to remove
            
        Returns:
            The removed goal
        
        Raises:
            IndexError: If the index is out of range
        """
        if not self.goals:
            raise IndexError("No goals to remove")
        
        return self.goals.pop(index)
    
    def update_goal(self, index: int, new_goal: str) -> None:
        """
        Update a goal at the specified index.
        
        Args:
            index: The index of the goal to update
            new_goal: The new goal statement
            
        Raises:
            IndexError: If the index is out of range
        """
        if index >= len(self.goals):
            raise IndexError(f"Goal index {index} out of range")
        
        self.goals[index] = new_goal
    
    def get_variable_type(self, name: str) -> Optional[str]:
        """
        Get the type of a variable.
        
        Args:
            name: The variable name
            
        Returns:
            The type of the variable or None if not found
        """
        return self.variables.get(name)
    
    def has_variable(self, name: str) -> bool:
        """
        Check if a variable exists in the context.
        
        Args:
            name: The variable name
            
        Returns:
            True if the variable exists, False otherwise
        """
        return name in self.variables
    
    def has_hypothesis(self, name: str) -> bool:
        """
        Check if a hypothesis exists in the context.
        
        Args:
            name: The hypothesis name
            
        Returns:
            True if the hypothesis exists, False otherwise
        """
        return name in self.hypotheses
    
    def has_definition(self, name: str) -> bool:
        """
        Check if a definition exists in the context.
        
        Args:
            name: The definition name
            
        Returns:
            True if the definition exists, False otherwise
        """
        return name in self.definitions
    
    def has_theorem(self, name: str) -> bool:
        """
        Check if a theorem exists in the context.
        
        Args:
            name: The theorem name
            
        Returns:
            True if the theorem exists, False otherwise
        """
        return name in self.theorems
    
    def has_import(self, import_name: str) -> bool:
        """
        Check if a library is imported.
        
        Args:
            import_name: The name of the imported library
            
        Returns:
            True if the library is imported, False otherwise
        """
        return import_name in self.imports
    
    def get_current_goal(self) -> Optional[str]:
        """
        Get the current goal.
        
        Returns:
            The current goal or None if there are no goals
        """
        return self.goals[0] if self.goals else None
    
    def get_all_goals(self) -> List[str]:
        """
        Get all goals.
        
        Returns:
            List of all goals
        """
        return self.goals
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata.
        
        Args:
            key: The metadata key
            default: Default value if key not found
            
        Returns:
            The metadata value or default if not found
        """
        return self.metadata.get(key, default)
    
    def clear(self) -> None:
        """Clear the proof context."""
        self.variables.clear()
        self.hypotheses.clear()
        self.definitions.clear()
        self.theorems.clear()
        self.imports.clear()
        self.goals.clear()
        self.metadata.clear()
    
    def clone(self) -> 'ProofContext':
        """
        Create a clone of this context.
        
        Returns:
            A new ProofContext instance with the same state
        """
        new_context = ProofContext()
        new_context.variables = self.variables.copy()
        new_context.hypotheses = self.hypotheses.copy()
        new_context.definitions = self.definitions.copy()
        new_context.theorems = self.theorems.copy()
        new_context.imports = self.imports.copy()
        new_context.goals = self.goals.copy()
        new_context.metadata = self.metadata.copy()
        return new_context
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "variables": self.variables,
            "hypotheses": self.hypotheses,
            "definitions": self.definitions,
            "theorems": self.theorems,
            "imports": list(self.imports),
            "goals": self.goals,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProofContext':
        """
        Create a context from a dictionary.
        
        Args:
            data: Dictionary representation of the context
            
        Returns:
            A new ProofContext instance
        """
        context = cls()
        context.variables = data.get("variables", {})
        context.hypotheses = data.get("hypotheses", {})
        context.definitions = data.get("definitions", {})
        context.theorems = data.get("theorems", {})
        context.imports = set(data.get("imports", []))
        context.goals = data.get("goals", [])
        context.metadata = data.get("metadata", {})
        return context
    
    def analyze_theorem_statement(self, statement: str) -> Dict[str, Any]:
        """
        Analyze a theorem statement to extract variables and structure.
        
        Args:
            statement: The theorem statement
            
        Returns:
            Dictionary with analysis information
        """
        result = {
            "variables": [],
            "predicates": [],
            "quantifiers": [],
            "symbols": [],
            "structure": "unknown"
        }
        
        # Extract variables (assuming single-letter variables for simplicity)
        variables = re.findall(r'\b([a-zA-Z])\b', statement)
        result["variables"] = list(set(variables))
        
        # Extract predicates (assuming predicate names are capitalized)
        predicates = re.findall(r'\b([A-Z][a-zA-Z]*)\b', statement)
        result["predicates"] = list(set(predicates))
        
        # Extract mathematical symbols
        symbols = re.findall(r'[+\-*/=<>∀∃∧∨¬⇒⇔]', statement)
        result["symbols"] = list(set(symbols))
        
        # Detect quantifiers
        if "forall" in statement or "∀" in statement:
            result["quantifiers"].append("universal")
        
        if "exists" in statement or "∃" in statement:
            result["quantifiers"].append("existential")
        
        # Try to categorize the theorem structure
        if "=" in statement or "equals" in statement:
            result["structure"] = "equality"
        elif ">" in statement or "<" in statement or "≤" in statement or "≥" in statement:
            result["structure"] = "inequality"
        elif "and" in statement or "∧" in statement:
            result["structure"] = "conjunction"
        elif "or" in statement or "∨" in statement:
            result["structure"] = "disjunction"
        elif "implies" in statement or "⇒" in statement:
            result["structure"] = "implication"
        elif "iff" in statement or "if and only if" in statement or "⇔" in statement:
            result["structure"] = "equivalence"
        
        return result


# Standalone functions for use in other modules

def create_proof_context() -> ProofContext:
    """
    Create a new proof context.
    
    Returns:
        A new ProofContext instance
    """
    return ProofContext()

def analyze_theorem(statement: str) -> Dict[str, Any]:
    """
    Analyze a theorem statement.
    
    Args:
        statement: The theorem statement
        
    Returns:
        Dictionary with analysis information
    """
    context = ProofContext()
    return context.analyze_theorem_statement(statement)

def create_context_from_variables(variables: Dict[str, str]) -> ProofContext:
    """
    Create a context with predefined variables.
    
    Args:
        variables: Dictionary mapping variable names to types
        
    Returns:
        A new ProofContext instance with the variables
    """
    context = ProofContext()
    for name, type_name in variables.items():
        context.add_variable(name, type_name)
    return context