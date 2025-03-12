"""
Concept indexing system for the knowledge base.
Provides efficient lookup and indexing of mathematical concepts.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger("knowledge_base")

class ConceptIndex:
    """
    Index for efficient lookup of mathematical concepts.
    
    This class provides methods for indexing and retrieving
    mathematical concepts from the knowledge base. It supports
    searching by keyword, domain, and concept name.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the concept index.
        
        Args:
            kb: Optional DomainKnowledgeBase instance
        """
        self.kb = kb
        self.index = {}
        self.domain_index = {}
        self.keyword_index = {}
        
        if kb:
            self.build_index()
    
    def build_index(self):
        """Build the concept index from the knowledge base."""
        logger.info("Building concept index...")
        
        # Clear existing indices
        self.index = {}
        self.domain_index = {}
        self.keyword_index = {}
        
        # Index all concepts
        for concept, info in self.kb.concepts.items():
            self.index[concept] = info
            
            # Index by domain
            for domain in info.get("domains", {}).keys():
                if domain not in self.domain_index:
                    self.domain_index[domain] = []
                self.domain_index[domain].append(concept)
            
            # Index by keyword
            keywords = self._extract_keywords(concept, info)
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(concept)
        
        logger.info(f"Concept index built with {len(self.index)} concepts, "
                   f"{len(self.domain_index)} domains, and {len(self.keyword_index)} keywords")
    
    def _extract_keywords(self, concept: str, info: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from a concept.
        
        Args:
            concept: The concept name
            info: The concept information
            
        Returns:
            List of keywords
        """
        keywords = [concept]
        
        # Add definition words
        if "definition" in info:
            definition = info["definition"].lower()
            definition_words = [word.strip(".,;:()[]{}") for word in definition.split() if len(word) > 3]
            keywords.extend(definition_words)
        
        # Add related concepts
        if "related_concepts" in info:
            keywords.extend(info["related_concepts"])
        
        # Add synonyms if available
        if "synonyms" in info:
            keywords.extend(info["synonyms"])
        
        return list(set(keywords))
    
    def search(self, query: str, domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Search for concepts matching a query.
        
        Args:
            query: The search query
            domain: Optional domain to restrict the search
            
        Returns:
            List of (concept, score) tuples
        """
        query = query.lower()
        results = []
        
        # Direct match
        if query in self.index:
            results.append((query, 1.0))
        
        # Search by keyword
        query_words = [word.strip(".,;:()[]{}") for word in query.split()]
        for word in query_words:
            if word in self.keyword_index:
                for concept in self.keyword_index[word]:
                    # Skip if domain is specified and concept is not in that domain
                    if domain and domain not in self.index[concept].get("domains", {}):
                        continue
                    
                    # Calculate score based on match quality
                    score = self._calculate_score(concept, query, word)
                    results.append((concept, score))
        
        # Remove duplicates and sort by score
        unique_results = {}
        for concept, score in results:
            if concept not in unique_results or score > unique_results[concept]:
                unique_results[concept] = score
        
        return sorted([(c, s) for c, s in unique_results.items()], key=lambda x: x[1], reverse=True)
    
    def _calculate_score(self, concept: str, query: str, matched_word: str) -> float:
        """
        Calculate a relevance score for a concept match.
        
        Args:
            concept: The matched concept
            query: The original query
            matched_word: The word that matched
            
        Returns:
            A relevance score between 0 and 1
        """
        # Exact concept match gets highest score
        if concept.lower() == query:
            return 1.0
        
        # Concept contains query as substring
        if query in concept.lower():
            return 0.9
        
        # Query contains concept as substring
        if concept.lower() in query:
            return 0.8
        
        # Matched word is the concept name
        if matched_word == concept.lower():
            return 0.7
        
        # Matched word is in the concept definition
        if "definition" in self.index[concept]:
            definition = self.index[concept]["definition"].lower()
            if matched_word in definition:
                return 0.6
        
        # Related concept match
        if "related_concepts" in self.index[concept]:
            related_concepts = [c.lower() for c in self.index[concept]["related_concepts"]]
            if matched_word in related_concepts:
                return 0.5
        
        # Default score for other matches
        return 0.3
    
    def get_concepts_by_domain(self, domain: str) -> List[str]:
        """
        Get all concepts in a domain.
        
        Args:
            domain: The domain code
            
        Returns:
            List of concepts in the domain
        """
        return self.domain_index.get(domain, [])
    
    def get_related_concepts(self, concept: str, max_depth: int = 1) -> List[str]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept: The concept name
            max_depth: Maximum depth for traversing relationships
            
        Returns:
            List of related concepts
        """
        if concept not in self.index:
            return []
        
        related = set()
        to_process = [(concept, 0)]
        processed = set()
        
        while to_process:
            current, depth = to_process.pop(0)
            
            if current in processed:
                continue
            
            processed.add(current)
            
            if depth > 0:  # Don't add the original concept
                related.add(current)
            
            if depth < max_depth and current in self.index:
                # Add direct relationships
                if "related_concepts" in self.index[current]:
                    for related_concept in self.index[current]["related_concepts"]:
                        if related_concept not in processed:
                            to_process.append((related_concept, depth + 1))
        
        return list(related)