"""
Knowledge base package for the proof translation system.
Provides domain-specific mathematical knowledge and library mappings.
"""

from knowledge.domain_kb import DomainKnowledgeBase
from knowledge.library_connectors import LibraryConnector, CoqLibraryConnector, LeanLibraryConnector
from knowledge.concept_index import ConceptIndex

__all__ = [
    'DomainKnowledgeBase',
    'LibraryConnector',
    'CoqLibraryConnector', 
    'LeanLibraryConnector',
    'ConceptIndex'
]