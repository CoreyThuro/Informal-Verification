"""
Knowledge Graph for Mathematical Concepts.

This module implements a simple knowledge graph for mathematical concepts,
focusing on relationships between mathematical entities, properties, and theorems.
It enhances the NLP capabilities by providing domain-specific understanding.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import json
import os
import re

class MathNode:
    """Represents a node in the mathematical knowledge graph."""
    
    def __init__(self, name: str, node_type: str, properties: Dict[str, Any] = None):
        """
        Initialize a mathematical concept node.
        
        Args:
            name: The name/identifier of the node
            node_type: The type of the node (concept, property, theorem, etc.)
            properties: Additional properties of the node
        """
        self.name = name
        self.node_type = node_type
        self.properties = properties or {}
        self.relationships = {}  # Maps relationship_type -> set of related node names
        
    def add_relationship(self, relationship_type: str, target_node: str):
        """
        Add a relationship from this node to another node.
        
        Args:
            relationship_type: The type of relationship
            target_node: The name of the target node
        """
        if relationship_type not in self.relationships:
            self.relationships[relationship_type] = set()
        self.relationships[relationship_type].add(target_node)
    
    def get_related_nodes(self, relationship_type: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Get nodes related to this node, optionally filtered by relationship type.
        
        Args:
            relationship_type: Optional filter for relationship type
            
        Returns:
            Dictionary mapping relationship types to sets of related node names
        """
        if relationship_type:
            return {relationship_type: self.relationships.get(relationship_type, set())}
        return self.relationships
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.node_type,
            "properties": self.properties,
            "relationships": {k: list(v) for k, v in self.relationships.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathNode':
        """Create a node from a dictionary representation."""
        node = cls(data["name"], data["type"], data["properties"])
        for rel_type, targets in data["relationships"].items():
            for target in targets:
                node.add_relationship(rel_type, target)
        return node


class MathKnowledgeGraph:
    """
    A knowledge graph for mathematical concepts.
    
    This graph represents mathematical concepts, their properties,
    and relationships between them to enhance semantic understanding.
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.nodes: Dict[str, MathNode] = {}
        self.node_types: Set[str] = set()
        self.relationship_types: Set[str] = set()
    
    def add_node(self, node: MathNode) -> None:
        """
        Add a node to the knowledge graph.
        
        Args:
            node: The node to add
        """
        self.nodes[node.name] = node
        self.node_types.add(node.node_type)
        for rel_type in node.relationships:
            self.relationship_types.add(rel_type)
    
    def get_node(self, name: str) -> Optional[MathNode]:
        """
        Get a node by name.
        
        Args:
            name: The name of the node to retrieve
            
        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(name)
    
    def add_relationship(self, source: str, relationship_type: str, target: str) -> bool:
        """
        Add a relationship between two nodes.
        
        Args:
            source: The name of the source node
            relationship_type: The type of relationship
            target: The name of the target node
            
        Returns:
            True if the relationship was added, False otherwise
        """
        if source not in self.nodes or target not in self.nodes:
            return False
        
        self.nodes[source].add_relationship(relationship_type, target)
        self.relationship_types.add(relationship_type)
        return True
    
    def get_related_nodes(self, node_name: str, relationship_type: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Get nodes related to a given node.
        
        Args:
            node_name: The name of the node
            relationship_type: Optional filter for relationship type
            
        Returns:
            Dictionary mapping relationship types to sets of related node names
        """
        node = self.get_node(node_name)
        if not node:
            return {}
        return node.get_related_nodes(relationship_type)
    
    def find_nodes_by_type(self, node_type: str) -> List[MathNode]:
        """
        Find all nodes of a given type.
        
        Args:
            node_type: The type of nodes to find
            
        Returns:
            List of nodes of the specified type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def find_path(self, source: str, target: str, max_depth: int = 3) -> List[Tuple[str, str, str]]:
        """
        Find a path between two nodes in the graph.
        
        Args:
            source: The name of the source node
            target: The name of the target node
            max_depth: Maximum path length to consider
            
        Returns:
            List of (node, relationship, node) tuples representing the path
        """
        if source not in self.nodes or target not in self.nodes:
            return []
        
        # Simple BFS to find a path
        visited = {source}
        queue = [(source, [])]
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.pop(0)
            
            for rel_type, targets in self.nodes[current].relationships.items():
                for next_node in targets:
                    if next_node == target:
                        # Found the target, return the path
                        return path + [(current, rel_type, next_node)]
                    
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [(current, rel_type, next_node)]))
        
        return []  # No path found
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> Dict[str, float]:
        """
        Get concepts related to a given concept with relevance scores.
        
        Args:
            concept: The concept to find related concepts for
            max_depth: Maximum relationship depth to consider
            
        Returns:
            Dictionary mapping related concept names to relevance scores
        """
        if concept not in self.nodes:
            # Try to find a similar concept
            similar_concepts = self.find_similar_concepts(concept)
            if not similar_concepts:
                return {}
            concept = similar_concepts[0][0]  # Use the most similar concept
        
        # BFS to find related concepts with decreasing relevance by depth
        related = {}
        visited = {concept}
        queue = [(concept, 1.0)]  # (node, relevance)
        
        while queue:
            current, relevance = queue.pop(0)
            
            if current != concept:  # Don't include the original concept
                related[current] = relevance
            
            if relevance < 0.2:  # Stop if relevance gets too low
                continue
            
            # Decrease relevance with each step
            next_relevance = relevance * 0.5
            
            for rel_type, targets in self.nodes[current].relationships.items():
                for next_node in targets:
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, next_relevance))
        
        return related
    
    def find_similar_concepts(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find concepts similar to the query string.
        
        Args:
            query: The query string
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (concept_name, similarity_score) tuples
        """
        # Simple string similarity for now
        query = query.lower()
        results = []
        
        for node_name in self.nodes:
            # Calculate simple string similarity
            name = node_name.lower()
            
            # Check for exact substring match
            if query in name or name in query:
                similarity = len(query) / max(len(query), len(name))
                if similarity >= threshold:
                    results.append((node_name, similarity))
                continue
            
            # Check word-level similarity
            query_words = set(query.split())
            name_words = set(name.split())
            
            if query_words and name_words:
                common_words = query_words.intersection(name_words)
                if common_words:
                    similarity = len(common_words) / max(len(query_words), len(name_words))
                    if similarity >= threshold:
                        results.append((node_name, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_concept_properties(self, concept: str) -> Dict[str, Any]:
        """
        Get the properties of a concept.
        
        Args:
            concept: The concept name
            
        Returns:
            Dictionary of property names to values
        """
        node = self.get_node(concept)
        if not node:
            return {}
        return node.properties
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            filepath: Path to save the file
        """
        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "node_types": list(self.node_types),
            "relationship_types": list(self.relationship_types)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MathKnowledgeGraph':
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded knowledge graph
        """
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        for node_data in data["nodes"]:
            graph.add_node(MathNode.from_dict(node_data))
        
        return graph


# Initialize with basic number theory concepts
def create_number_theory_graph() -> MathKnowledgeGraph:
    """
    Create a knowledge graph with basic number theory concepts.
    
    Returns:
        A knowledge graph populated with number theory concepts
    """
    graph = MathKnowledgeGraph()
    
    # Add basic number concepts
    graph.add_node(MathNode("integer", "concept", {"definition": "A whole number that can be positive, negative, or zero"}))
    graph.add_node(MathNode("natural_number", "concept", {"definition": "A positive integer"}))
    graph.add_node(MathNode("even_number", "concept", {"definition": "An integer divisible by 2"}))
    graph.add_node(MathNode("odd_number", "concept", {"definition": "An integer not divisible by 2"}))
    graph.add_node(MathNode("prime_number", "concept", {"definition": "A natural number greater than 1 that is not a product of two smaller natural numbers"}))
    graph.add_node(MathNode("composite_number", "concept", {"definition": "A natural number greater than 1 that has positive divisors other than 1 and itself"}))
    
    # Add properties
    graph.add_node(MathNode("divisibility", "property", {"definition": "A number divides another number without a remainder"}))
    graph.add_node(MathNode("parity", "property", {"definition": "Whether a number is even or odd"}))
    graph.add_node(MathNode("primality", "property", {"definition": "Whether a number is prime"}))
    
    # Add operations
    graph.add_node(MathNode("addition", "operation", {"symbol": "+", "definition": "Combining two numbers to get their sum"}))
    graph.add_node(MathNode("multiplication", "operation", {"symbol": "×", "definition": "Repeated addition of a number"}))
    graph.add_node(MathNode("division", "operation", {"symbol": "÷", "definition": "Finding how many times one number is contained in another"}))
    graph.add_node(MathNode("modulo", "operation", {"symbol": "%", "definition": "Finding the remainder after division"}))
    
    # Add theorems
    graph.add_node(MathNode("evenness_theorem", "theorem", {
        "statement": "A number is even if and only if it is divisible by 2",
        "proof_patterns": ["evenness", "direct"]
    }))
    graph.add_node(MathNode("odd_sum_theorem", "theorem", {
        "statement": "The sum of two odd numbers is even",
        "proof_patterns": ["evenness", "direct"]
    }))
    graph.add_node(MathNode("even_product_theorem", "theorem", {
        "statement": "If a number is even, then its product with any integer is even",
        "proof_patterns": ["evenness", "direct"]
    }))
    graph.add_node(MathNode("division_theorem", "theorem", {
        "statement": "For any integers a and b with b ≠ 0, there exist unique integers q and r such that a = bq + r and 0 ≤ r < |b|",
        "proof_patterns": ["cases", "induction"]
    }))
    
    # Add relationships
    graph.add_relationship("natural_number", "is_a", "integer")
    graph.add_relationship("even_number", "is_a", "integer")
    graph.add_relationship("odd_number", "is_a", "integer")
    graph.add_relationship("prime_number", "is_a", "natural_number")
    graph.add_relationship("composite_number", "is_a", "natural_number")
    
    graph.add_relationship("even_number", "has_property", "parity")
    graph.add_relationship("odd_number", "has_property", "parity")
    graph.add_relationship("prime_number", "has_property", "primality")
    graph.add_relationship("composite_number", "has_property", "primality")
    
    graph.add_relationship("even_number", "related_to", "divisibility")
    graph.add_relationship("divisibility", "related_to", "division")
    graph.add_relationship("divisibility", "related_to", "modulo")
    
    graph.add_relationship("evenness_theorem", "involves", "even_number")
    graph.add_relationship("evenness_theorem", "involves", "divisibility")
    graph.add_relationship("odd_sum_theorem", "involves", "odd_number")
    graph.add_relationship("odd_sum_theorem", "involves", "even_number")
    graph.add_relationship("odd_sum_theorem", "involves", "addition")
    graph.add_relationship("even_product_theorem", "involves", "even_number")
    graph.add_relationship("even_product_theorem", "involves", "multiplication")
    graph.add_relationship("division_theorem", "involves", "division")
    graph.add_relationship("division_theorem", "involves", "modulo")
    
    return graph


# Create a default graph instance
default_graph = create_number_theory_graph()
