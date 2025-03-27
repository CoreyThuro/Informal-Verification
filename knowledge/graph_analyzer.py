"""
Knowledge Graph Analyzer for Mathematical Proofs.

This module integrates the knowledge graph with the NLP analyzer
to enhance pattern recognition and semantic understanding of mathematical proofs.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import re
import spacy
from spacy.tokens import Doc, Token, Span

import spacy
from patterns.nlp_analyzer import analyze_proof, get_enhanced_pattern
from knowledge.knowledge_graph import MathKnowledgeGraph, default_graph

class GraphEnhancedAnalyzer:
    """
    Analyzer that uses a knowledge graph to enhance NLP analysis of mathematical proofs.
    
    This class extends the capabilities of the NLP analyzer functions by incorporating
    domain knowledge from a mathematical knowledge graph.
    """
    
    def __init__(self, knowledge_graph: Optional[MathKnowledgeGraph] = None):
        """
        Initialize the graph-enhanced analyzer.
        
        Args:
            knowledge_graph: Optional knowledge graph (uses default if not provided)
        """
        self.knowledge_graph = knowledge_graph or default_graph
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If the model isn't installed, download it
            import subprocess
            import sys
            print("Downloading spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def identify_math_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify mathematical concepts in the text using the knowledge graph.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of identified concepts with their positions and relevance
        """
        doc = self.nlp(text)
        concepts = []
        
        # Extract noun phrases and check against knowledge graph
        for chunk in doc.noun_chunks:
            # Check for exact matches
            concept_name = chunk.text.lower()
            node = self.knowledge_graph.get_node(concept_name)
            
            if not node:
                # Try with underscores instead of spaces
                concept_name = concept_name.replace(" ", "_")
                node = self.knowledge_graph.get_node(concept_name)
            
            if not node:
                # Try to find similar concepts
                similar = self.knowledge_graph.find_similar_concepts(chunk.text)
                if similar:
                    node = self.knowledge_graph.get_node(similar[0][0])
                    similarity = similar[0][1]
                else:
                    continue
            else:
                similarity = 1.0
            
            if node:
                concepts.append({
                    "text": chunk.text,
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                    "concept": node.name,
                    "type": node.node_type,
                    "similarity": similarity,
                    "properties": node.properties
                })
        
        # Also check individual tokens for mathematical terms
        for token in doc:
            # Skip tokens that are already part of identified concepts
            if any(c["start"] <= token.idx < c["end"] for c in concepts):
                continue
            
            concept_name = token.text.lower()
            node = self.knowledge_graph.get_node(concept_name)
            
            if node:
                concepts.append({
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "concept": node.name,
                    "type": node.node_type,
                    "similarity": 1.0,
                    "properties": node.properties
                })
        
        return concepts
    
    def get_related_concepts(self, concept_names: List[str]) -> Dict[str, float]:
        """
        Get concepts related to the given concepts with relevance scores.
        
        Args:
            concept_names: List of concept names to find related concepts for
            
        Returns:
            Dictionary mapping related concept names to relevance scores
        """
        related = {}
        
        for name in concept_names:
            # Get related concepts for this concept
            concept_related = self.knowledge_graph.get_related_concepts(name)
            
            # Merge with existing related concepts, taking the maximum relevance
            for concept, relevance in concept_related.items():
                related[concept] = max(related.get(concept, 0), relevance)
        
        return related
    
    def enhance_pattern_confidence(self, pattern: str, confidence: float, 
                                  theorem_text: str, proof_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Enhance pattern confidence using knowledge graph information.
        
        Args:
            pattern: The identified pattern
            confidence: The initial confidence score
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Enhanced confidence score and additional information
        """
        # Identify concepts in the theorem and proof
        theorem_concepts = self.identify_math_concepts(theorem_text)
        proof_concepts = self.identify_math_concepts(proof_text)
        
        # Get all concept names
        concept_names = [c["concept"] for c in theorem_concepts + proof_concepts]
        
        # Find theorems in the knowledge graph that relate to this pattern
        pattern_theorems = []
        for theorem_node in self.knowledge_graph.find_nodes_by_type("theorem"):
            if "proof_patterns" in theorem_node.properties:
                if pattern.lower() in [p.lower() for p in theorem_node.properties["proof_patterns"]]:
                    pattern_theorems.append(theorem_node.name)
        
        # Check if the identified concepts are related to the pattern theorems
        pattern_relevance = 0.0
        related_theorems = []
        
        for theorem_name in pattern_theorems:
            # Get concepts related to this theorem
            theorem_node = self.knowledge_graph.get_node(theorem_name)
            if not theorem_node:
                continue
                
            related = self.knowledge_graph.get_related_nodes(theorem_name, "involves")
            related_concepts = related.get("involves", set())
            
            # Check overlap with identified concepts
            overlap = [c for c in concept_names if c in related_concepts]
            
            if overlap:
                relevance = len(overlap) / len(related_concepts) if related_concepts else 0
                pattern_relevance = max(pattern_relevance, relevance)
                related_theorems.append({
                    "theorem": theorem_name,
                    "statement": theorem_node.properties.get("statement", ""),
                    "relevance": relevance,
                    "common_concepts": overlap
                })
        
        # Adjust confidence based on pattern relevance
        if pattern_relevance > 0:
            # Boost confidence based on knowledge graph relevance
            enhanced_confidence = confidence * (1 + pattern_relevance * 0.5)
            # Cap at 1.0
            enhanced_confidence = min(enhanced_confidence, 1.0)
        else:
            enhanced_confidence = confidence
        
        # Prepare additional information
        additional_info = {
            "theorem_concepts": theorem_concepts,
            "proof_concepts": proof_concepts,
            "related_theorems": related_theorems,
            "pattern_relevance": pattern_relevance
        }
        
        return enhanced_confidence, additional_info
    
    def analyze_proof(self, theorem_text: str, proof_text: str) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a theorem and its proof using the knowledge graph.
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            Dictionary with analysis results
        """
        # Get base NLP analysis
        base_analysis = analyze_proof(theorem_text, proof_text)
        
        # Get enhanced pattern recognition
        try:
            pattern, pattern_info = get_enhanced_pattern(theorem_text, proof_text)
            confidence = pattern_info.get('nlp_confidence', 0.5)  # Use NLP confidence if available
        except Exception as e:
            print(f"Warning: Enhanced pattern recognition failed: {e}")
            # Use the top pattern from pattern scores
            pattern_scores = base_analysis.get('pattern_scores', {})
            if pattern_scores:
                pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
                confidence = pattern_scores[pattern] / sum(pattern_scores.values()) if sum(pattern_scores.values()) > 0 else 0.0
            else:
                pattern = "unknown"
                confidence = 0.0
        
        # Identify mathematical concepts
        theorem_concepts = self.identify_math_concepts(theorem_text)
        proof_concepts = self.identify_math_concepts(proof_text)
        
        # Get all concept names
        concept_names = list(set([c["concept"] for c in theorem_concepts + proof_concepts]))
        
        # Get related concepts
        related_concepts = self.get_related_concepts(concept_names)
        
        # Enhance pattern confidence
        enhanced_confidence, additional_info = self.enhance_pattern_confidence(
            pattern, confidence, theorem_text, proof_text
        )
        
        # Prepare enhanced analysis
        enhanced_analysis = {
            **base_analysis,
            "pattern": pattern,
            "confidence": enhanced_confidence,
            "theorem_concepts": theorem_concepts,
            "proof_concepts": proof_concepts,
            "related_concepts": [{"name": name, "relevance": score} 
                                for name, score in related_concepts.items()],
            "knowledge_graph_info": additional_info
        }
        
        return enhanced_analysis
    
    def suggest_related_theorems(self, theorem_text: str, proof_text: str) -> List[Dict[str, Any]]:
        """
        Suggest theorems related to the given theorem and proof.
        
        Args:
            theorem_text: The theorem text
            proof_text: The proof text
            
        Returns:
            List of related theorems with relevance scores
        """
        # Identify concepts in the theorem and proof
        theorem_concepts = self.identify_math_concepts(theorem_text)
        proof_concepts = self.identify_math_concepts(proof_text)
        
        # Get all concept names
        concept_names = list(set([c["concept"] for c in theorem_concepts + proof_concepts]))
        
        # Find theorems that involve these concepts
        related_theorems = []
        for theorem_node in self.knowledge_graph.find_nodes_by_type("theorem"):
            # Get concepts related to this theorem
            related = self.knowledge_graph.get_related_nodes(theorem_node.name, "involves")
            related_concepts = related.get("involves", set())
            
            # Check overlap with identified concepts
            overlap = [c for c in concept_names if c in related_concepts]
            
            if overlap:
                relevance = len(overlap) / len(related_concepts) if related_concepts else 0
                related_theorems.append({
                    "theorem": theorem_node.name,
                    "statement": theorem_node.properties.get("statement", ""),
                    "relevance": relevance,
                    "common_concepts": overlap
                })
        
        # Sort by relevance
        related_theorems.sort(key=lambda x: x["relevance"], reverse=True)
        
        return related_theorems
    
    def get_concept_explanations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical concepts from text and provide explanations.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of concepts with explanations
        """
        concepts = self.identify_math_concepts(text)
        
        # Add explanations from the knowledge graph
        for concept in concepts:
            node = self.knowledge_graph.get_node(concept["concept"])
            if node:
                # Get definition if available
                concept["explanation"] = node.properties.get("definition", "")
                
                # Get related concepts
                related = self.knowledge_graph.get_related_nodes(node.name)
                concept["relationships"] = {
                    rel_type: list(nodes) 
                    for rel_type, nodes in related.items()
                }
        
        return concepts
