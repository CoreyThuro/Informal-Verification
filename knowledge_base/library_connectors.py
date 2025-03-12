"""
Library connectors for theorem prover libraries.
Extracts information from Coq and Lean libraries.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger("knowledge_base")

class LibraryConnector:
    """
    Base class for connecting to theorem prover libraries.
    
    This class provides methods for extracting information from
    theorem prover library files, including definitions, theorems,
    and proof tactics.
    """
    
    def __init__(self, library_path: str):
        """
        Initialize the library connector.
        
        Args:
            library_path: Path to the library directory
        """
        self.library_path = library_path
        self.cache = {}
    
    def extract_concepts(self) -> Dict[str, Any]:
        """
        Extract concepts and their definitions from libraries.
        
        Returns:
            Dictionary mapping concept names to information
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement extract_concepts()")
    
    def get_dependencies(self, concept: str) -> List[str]:
        """
        Get dependencies for a concept.
        
        Args:
            concept: The concept name
            
        Returns:
            List of dependencies
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_dependencies()")


class CoqLibraryConnector(LibraryConnector):
    """
    Connector for Coq libraries.
    
    This class extracts information from Coq library files,
    including definitions, theorems, and tactics.
    """
    
    def extract_concepts(self) -> Dict[str, Any]:
        """
        Extract concepts from Coq libraries.
        
        Returns:
            Dictionary mapping concept names to information
        """
        logger.info(f"Extracting concepts from Coq libraries at: {self.library_path}")
        concepts = {}
        
        # Traverse the library directory
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith('.v'):
                    file_path = os.path.join(root, file)
                    try:
                        file_concepts = self._extract_concepts_from_file(file_path)
                        concepts.update(file_concepts)
                        logger.debug(f"Extracted {len(file_concepts)} concepts from {file_path}")
                    except Exception as e:
                        logger.warning(f"Error extracting concepts from {file_path}: {e}")
        
        return concepts
    
    def _extract_concepts_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract concepts from a single Coq file.
        
        Args:
            file_path: Path to the Coq file
            
        Returns:
            Dictionary mapping concept names to information
        """
        concepts = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return concepts
        
        # Extract module name from file path
        module_parts = []
        path_parts = file_path.replace(self.library_path, '').strip(os.sep).split(os.sep)
        for part in path_parts:
            if part.endswith('.v'):
                module_parts.append(part[:-2])  # Remove .v extension
            else:
                module_parts.append(part)
        module_name = '.'.join(module_parts)
        
        # Extract inductive definitions
        inductive_matches = re.finditer(r'Inductive\s+(\w+)', content)
        for match in inductive_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'inductive',
                'file': file_path,
                'module': module_name,
                'libraries': [f"Require Import {module_name}."]
            }
        
        # Extract definitions
        definition_matches = re.finditer(r'Definition\s+(\w+)', content)
        for match in definition_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'definition',
                'file': file_path,
                'module': module_name,
                'libraries': [f"Require Import {module_name}."]
            }
        
        # Extract theorems
        theorem_matches = re.finditer(r'Theorem\s+(\w+)', content)
        for match in theorem_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'theorem',
                'file': file_path,
                'module': module_name,
                'libraries': [f"Require Import {module_name}."]
            }
        
        # Extract lemmas
        lemma_matches = re.finditer(r'Lemma\s+(\w+)', content)
        for match in lemma_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'lemma',
                'file': file_path,
                'module': module_name,
                'libraries': [f"Require Import {module_name}."]
            }
        
        return concepts
    
    def get_dependencies(self, concept: str) -> List[str]:
        """
        Get dependencies for a concept in Coq.
        
        Args:
            concept: The concept name
            
        Returns:
            List of dependencies
        """
        # Check if the concept is in the cache
        if concept in self.cache:
            return self.cache[concept]
        
        # Find the concept in the extracted concepts
        concepts = self.extract_concepts()
        if concept not in concepts:
            return []
        
        # Get the file path
        file_path = concepts[concept].get('file')
        if not file_path:
            return []
        
        # Parse the file to find dependencies
        dependencies = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return dependencies
        
        # Extract imports
        import_matches = re.finditer(r'Require Import\s+([^.]+)\.', content)
        for match in import_matches:
            dependency = match.group(1)
            dependencies.append(dependency)
        
        # Cache the dependencies
        self.cache[concept] = dependencies
        
        return dependencies


class LeanLibraryConnector(LibraryConnector):
    """
    Connector for Lean libraries.
    
    This class extracts information from Lean library files,
    including definitions, theorems, and tactics.
    """
    
    def extract_concepts(self) -> Dict[str, Any]:
        """
        Extract concepts from Lean libraries.
        
        Returns:
            Dictionary mapping concept names to information
        """
        logger.info(f"Extracting concepts from Lean libraries at: {self.library_path}")
        concepts = {}
        
        # Traverse the library directory
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith('.lean'):
                    file_path = os.path.join(root, file)
                    try:
                        file_concepts = self._extract_concepts_from_file(file_path)
                        concepts.update(file_concepts)
                        logger.debug(f"Extracted {len(file_concepts)} concepts from {file_path}")
                    except Exception as e:
                        logger.warning(f"Error extracting concepts from {file_path}: {e}")
        
        return concepts
    
    def _extract_concepts_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract concepts from a single Lean file.
        
        Args:
            file_path: Path to the Lean file
            
        Returns:
            Dictionary mapping concept names to information
        """
        concepts = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return concepts
        
        # Extract module name from file path
        module_parts = []
        path_parts = file_path.replace(self.library_path, '').strip(os.sep).split(os.sep)
        for part in path_parts:
            if part.endswith('.lean'):
                module_parts.append(part[:-5])  # Remove .lean extension
            else:
                module_parts.append(part)
        module_name = '.'.join(module_parts)
        
        # Extract inductive definitions
        inductive_matches = re.finditer(r'inductive\s+(\w+)', content)
        for match in inductive_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'inductive',
                'file': file_path,
                'module': module_name,
                'libraries': [f"import {module_name}"]
            }
        
        # Extract definitions
        definition_matches = re.finditer(r'def\s+(\w+)', content)
        for match in definition_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'definition',
                'file': file_path,
                'module': module_name,
                'libraries': [f"import {module_name}"]
            }
        
        # Extract theorems
        theorem_matches = re.finditer(r'theorem\s+(\w+)', content)
        for match in theorem_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'theorem',
                'file': file_path,
                'module': module_name,
                'libraries': [f"import {module_name}"]
            }
        
        # Extract lemmas
        lemma_matches = re.finditer(r'lemma\s+(\w+)', content)
        for match in lemma_matches:
            name = match.group(1)
            concepts[name] = {
                'type': 'lemma',
                'file': file_path,
                'module': module_name,
                'libraries': [f"import {module_name}"]
            }
        
        return concepts
    
    def get_dependencies(self, concept: str) -> List[str]:
        """
        Get dependencies for a concept in Lean.
        
        Args:
            concept: The concept name
            
        Returns:
            List of dependencies
        """
        # Check if the concept is in the cache
        if concept in self.cache:
            return self.cache[concept]
        
        # Find the concept in the extracted concepts
        concepts = self.extract_concepts()
        if concept not in concepts:
            return []
        
        # Get the file path
        file_path = concepts[concept].get('file')
        if not file_path:
            return []
        
        # Parse the file to find dependencies
        dependencies = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return dependencies
        
        # Extract imports
        import_matches = re.finditer(r'import\s+([^\n]+)', content)
        for match in import_matches:
            dependency = match.group(1)
            dependencies.append(dependency)
        
        # Cache the dependencies
        self.cache[concept] = dependencies
        
        return dependencies