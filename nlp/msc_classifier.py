"""
Mathematics Subject Classification (MSC) based domain classifier.
Classifies mathematical texts into appropriate MSC categories.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re

# Configure logging
logger = logging.getLogger("msc_classifier")

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("SentenceTransformer not available. Using fallback approach.")
    HAS_SENTENCE_TRANSFORMERS = False

class MSCClassifier:
    """
    Classifier for mathematical domains based on the Mathematics Subject Classification (MSC).
    """
    
    def __init__(self, kb=None, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the MSC classifier.
        
        Args:
            kb: Optional DomainKnowledgeBase instance
            model_name: Name of the sentence transformer model to use
        """
        self.kb = kb
        self.model_name = model_name
        
        # Dictionary of MSC codes and their descriptions
        self.msc_codes = self._load_msc_codes()
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.msc_embeddings = {}
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self._compute_msc_embeddings()
                logger.info(f"Initialized embedding model {model_name}")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {e}")
                self.embedding_model = None
        
        # Initialize keyword-based approach as fallback
        self.domain_keywords = self._initialize_domain_keywords()
    
    def _load_msc_codes(self) -> Dict[str, Dict[str, str]]:
        """
        Load MSC codes and their descriptions.
        
        Returns:
            Dictionary mapping MSC codes to their information
        """
        if self.kb:
            # Get MSC codes from knowledge base
            return getattr(self.kb, 'domains', {})
        else:
            # Default MSC codes if knowledge base is not available
            return {
                "00": {"name": "General Mathematics", "description": "General mathematical topics"},
                "11": {"name": "Number Theory", "description": "Number theory, including divisibility, congruences, and primality"},
                "12": {"name": "Field Theory and Polynomials", "description": "Field theory and polynomials"},
                "13": {"name": "Commutative Algebra", "description": "Commutative algebra, rings, and ideals"},
                "14": {"name": "Algebraic Geometry", "description": "Algebraic geometry"},
                "15": {"name": "Linear Algebra", "description": "Linear and multilinear algebra, matrix theory"},
                "20": {"name": "Group Theory", "description": "Group theory and generalizations"},
                "26": {"name": "Real Functions", "description": "Real functions, analysis"},
                "30": {"name": "Functions of a Complex Variable", "description": "Complex analysis"},
                "35": {"name": "Partial Differential Equations", "description": "Partial differential equations"},
                "40": {"name": "Sequences, Series, Summability", "description": "Sequences, series, and summability"},
                "54": {"name": "General Topology", "description": "General topology"},
                "55": {"name": "Algebraic Topology", "description": "Algebraic topology"}
            }
    
    def _compute_msc_embeddings(self) -> None:
        """Compute embeddings for MSC categories."""
        if not self.embedding_model:
            return
        
        embed_texts = {}
        for code, info in self.msc_codes.items():
            name = info.get("name", "")
            description = info.get("description", "")
            # Create text for embedding
            embed_text = f"{name}. {description}"
            embed_texts[code] = embed_text
        
        # Compute embeddings in batch
        try:
            sentences = list(embed_texts.values())
            embeddings = self.embedding_model.encode(sentences)
            
            # Store embeddings for each MSC code
            for i, code in enumerate(embed_texts.keys()):
                self.msc_embeddings[code] = embeddings[i]
            
            logger.info(f"Computed embeddings for {len(self.msc_embeddings)} MSC categories")
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """
        Initialize domain keywords for fallback classification.
        
        Returns:
            Dictionary mapping domain codes to keywords
        """
        # This is a fallback if embeddings are not available
        return {
            "11": [  # Number Theory
                "prime", "number", "integer", "divisor", "divisible", "modulo", "congruence", 
                "gcd", "lcm", "factor", "remainder", "quotient", "coprime", "prime", 
                "composite", "even", "odd", "natural", "divisibility", "primality", "parity"
            ],
            "12-15": [  # Algebra
                "group", "field", "ring", "algebra", "module", "polynomial", "matrix", 
                "determinant", "eigenvalue", "vector", "basis", "linear", "algebra", 
                "homomorphism", "isomorphism", "subgroup", "permutation", "symmetry",
                "linear equation", "linear system", "column", "row", "span", "dimension"
            ],
            "26-42": [  # Analysis
                "limit", "continuity", "differentiable", "integral", "derivative", "series", 
                "sequence", "convergence", "function", "real", "complex", "measure", 
                "bounded", "continuous", "uniformly", "supremum", "infimum", "epsilon", "delta",
                "interval", "neighborhood", "absolute", "converge"
            ],
            "54-55": [  # Topology
                "open", "closed", "compact", "connected", "hausdorff", "neighborhood", 
                "continuous", "homeomorphism", "topology", "metric", "space", "cover", 
                "manifold", "boundary", "interior", "closure", "dense", "separable",
                "topological", "compactness", "connectedness"
            ]
        }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a text into MSC categories.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Try embedding-based classification first
        if self.embedding_model and self.msc_embeddings:
            return self._classify_with_embeddings(text)
        else:
            # Fall back to keyword-based classification
            return self._classify_with_keywords(text)
    
    def _classify_with_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Classify text using embeddings.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Compute embedding for the text
        try:
            text_embedding = self.embedding_model.encode(text)
            
            # Calculate cosine similarity with MSC category embeddings
            similarities = {}
            for msc_code, embedding in self.msc_embeddings.items():
                similarity = self._cosine_similarity(text_embedding, embedding)
                similarities[msc_code] = similarity
            
            # Get top matches
            top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get detailed information for top matches
            results = []
            for msc_code, similarity in top_matches:
                msc_info = self.msc_codes.get(msc_code, {})
                results.append({
                    "msc_code": msc_code,
                    "name": msc_info.get("name", "Unknown"),
                    "confidence": float(similarity),
                    "description": msc_info.get("description", "")
                })
            
            return {
                "primary_domain": results[0]["msc_code"] if results else None,
                "confidence": results[0]["confidence"] if results else 0.0,
                "method": "embedding",
                "all_matches": results
            }
        except Exception as e:
            logger.error(f"Error in embedding classification: {e}")
            # Fall back to keyword-based classification
            return self._classify_with_keywords(text)
    
    def _classify_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Classify text using keywords.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Normalize text
        text = text.lower()
        
        # Count domain keywords
        domain_scores = {}
        keyword_matches = {}
        
        for domain, keywords in self.domain_keywords.items():
            domain_scores[domain] = 0
            keyword_matches[domain] = []
            
            for keyword in keywords:
                # Count occurrences of each keyword
                matches = re.findall(r'\b' + re.escape(keyword) + r'\b', text)
                if matches:
                    domain_scores[domain] += len(matches)
                    keyword_matches[domain].append((keyword, len(matches)))
        
        # Find domain with highest score
        max_score = max(domain_scores.values())
        top_domains = sorted(
            [(d, s) for d, s in domain_scores.items() if s > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert domain codes to standard MSC codes
        results = []
        for domain, score in top_domains[:3]:
            if "-" in domain:
                # Range of codes
                primary_code = domain.split("-")[0]
            else:
                primary_code = domain
                
            msc_info = self.msc_codes.get(primary_code, {})
            confidence = score / sum(domain_scores.values()) if sum(domain_scores.values()) > 0 else 0.0
            
            results.append({
                "msc_code": primary_code,
                "name": msc_info.get("name", "Unknown"),
                "confidence": confidence,
                "description": msc_info.get("description", ""),
                "keyword_matches": keyword_matches[domain]
            })
        
        return {
            "primary_domain": results[0]["msc_code"] if results else None,
            "confidence": results[0]["confidence"] if results else 0.0,
            "method": "keywords",
            "all_matches": results
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_detailed_classification(self, text: str) -> Dict[str, Any]:
        """
        Get detailed classification with concept extraction.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with detailed classification results
        """
        # Get basic classification
        basic_result = self.classify(text)
        
        # Extract mathematical concepts
        concepts = self._extract_mathematical_concepts(text, basic_result.get("primary_domain"))
        
        # Get domain information
        domain_code = basic_result.get("primary_domain")
        if self.kb and domain_code:
            domain_info = self.kb.get_domain_info(domain_code)
        else:
            domain_info = self.msc_codes.get(domain_code, {})
        
        return {
            "primary_domain": basic_result.get("primary_domain"),
            "confidence": basic_result.get("confidence"),
            "method": basic_result.get("method"),
            "all_matches": basic_result.get("all_matches", []),
            "domain_info": domain_info,
            "concepts": concepts
        }
    
    def _extract_mathematical_concepts(self, text: str, domain_code: Optional[str]) -> List[str]:
        """
        Extract mathematical concepts from text.
        
        Args:
            text: The text to analyze
            domain_code: The domain code for context
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Look for domain-specific concepts
        if domain_code and domain_code in self.domain_keywords:
            for keyword in self.domain_keywords[domain_code]:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    concepts.append(keyword)
        
        # Extract variable names (single letters, especially x, y, z, n, m, k)
        variables = re.findall(r'\b([a-zA-Z])\b', text)
        var_counts = {}
        for var in variables:
            var_counts[var] = var_counts.get(var, 0) + 1
        
        # Only add variables that appear multiple times
        for var, count in var_counts.items():
            if count > 1:
                concepts.append(f"variable:{var}")
        
        # Extract mathematical expressions
        # Basic expressions like "x + y", "n^2", etc.
        expressions = re.findall(r'\b([a-zA-Z])(?:\s*[\+\-\*\/\^]\s*([a-zA-Z0-9]+))+\b', text)
        for expr in expressions:
            if isinstance(expr, tuple):
                concepts.append(f"expression:{expr[0]}")
        
        # Extract function applications like "f(x)"
        functions = re.findall(r'\b([a-zA-Z])\s*\(\s*([a-zA-Z](?:,\s*[a-zA-Z])*)\s*\)\b', text)
        for func, args in functions:
            concepts.append(f"function:{func}")
        
        return list(set(concepts))

def classify_mathematical_domain(text: str, kb=None) -> Dict[str, Any]:
    """
    Classify a mathematical text into an MSC domain.
    
    Args:
        text: The text to classify
        kb: Optional knowledge base
        
    Returns:
        Dictionary with classification results
    """
    classifier = MSCClassifier(kb)
    return classifier.get_detailed_classification(text)