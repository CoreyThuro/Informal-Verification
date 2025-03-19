"""
Knowledge base package for the proof translation system.
Provides domain-specific mathematical knowledge and library mappings.
"""

from knowledge_base.simple_kb import SimpleKnowledgeBase


__all__ = [
    'SimpleKnowledgeBase',
    'CoqLibraryConnector', 
    'LeanLibraryConnector',
    'ConceptIndex'
]