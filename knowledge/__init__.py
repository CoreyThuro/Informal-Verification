"""
Knowledge base package for the proof translation system.
Provides domain-specific mathematical knowledge, library mappings,
and a knowledge graph for enhanced semantic understanding.
"""

from knowledge.kb import KnowledgeBase
from knowledge.knowledge_graph import MathKnowledgeGraph, MathNode, default_graph
from knowledge.graph_analyzer import GraphEnhancedAnalyzer

__all__ = [
    'KnowledgeBase',
    'MathKnowledgeGraph',
    'MathNode',
    'GraphEnhancedAnalyzer',
    'default_graph',
]