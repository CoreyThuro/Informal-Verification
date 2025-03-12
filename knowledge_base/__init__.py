"""
Knowledge base package for the proof translation system.
Provides domain-specific mathematical knowledge and library mappings.
"""

from knowledge_base.domain_kb import DomainKnowledgeBase
from knowledge_base.library_connectors import LibraryConnector, CoqLibraryConnector, LeanLibraryConnector
from knowledge_base.concept_index import ConceptIndex

__all__ = [
    'DomainKnowledgeBase',
    'LibraryConnector',
    'CoqLibraryConnector', 
    'LeanLibraryConnector',
    'ConceptIndex'
]