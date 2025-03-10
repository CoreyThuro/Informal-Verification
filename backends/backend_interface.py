"""
Interface definition for theorem prover backends.
Provides a common abstraction for different proof assistants.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import subprocess
import tempfile

from ir.proof_ir import ProofIR, TacticType

class ProverBackend(ABC):
    """
    Abstract base class for all theorem prover backends.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the theorem prover."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the theorem prover."""
        pass
    
    @property
    @abstractmethod
    def supported_libraries(self) -> List[str]:
        """List of supported libraries in this prover."""
        pass
    
    @abstractmethod
    def translate(self, proof_ir: ProofIR) -> str:
        """
        Translate the proof IR into prover-specific syntax.
        
        Args:
            proof_ir: The intermediate representation of the proof
            
        Returns:
            The proof script in the prover's syntax
        """
        pass
    
    @abstractmethod
    def verify(self, proof_script: str, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify the proof script with the prover.
        
        Args:
            proof_script: The proof script to verify
            filename: Optional filename to save the script to
            
        Returns:
            Tuple of (verification success, error message if any)
        """
        pass
    
    @abstractmethod
    def map_concept(self, concept: str, domain: Optional[str] = None) -> str:
        """
        Map a mathematical concept to its prover-specific representation.
        
        Args:
            concept: The concept to map
            domain: Optional domain for context
            
        Returns:
            The prover-specific representation
        """
        pass
    
    @abstractmethod
    def map_tactic(self, tactic_type: TacticType, args: List[Any] = None) -> str:
        """
        Map a generic tactic to its prover-specific syntax.
        
        Args:
            tactic_type: The type of tactic
            args: Optional arguments for the tactic
            
        Returns:
            The prover-specific tactic syntax
        """
        pass
    
    @abstractmethod
    def interactive_session(self, proof_script: str) -> str:
        """
        Start an interactive session with this prover.
        
        Args:
            proof_script: The proof script to load
            
        Returns:
            Path to a temporary file containing the proof script
        """
        pass
    
    def is_installed(self) -> bool:
        """
        Check if this prover is installed on the system.
        
        Returns:
            True if installed, False otherwise
        """
        try:
            self._check_installation()
            return True
        except Exception:
            return False
    
    @abstractmethod
    def _check_installation(self) -> None:
        """
        Implementation-specific installation check.
        Raises an exception if the prover is not installed.
        """
        pass
    
    def process_feedback(self, error_message: str) -> Dict[str, Any]:
        """
        Process error feedback from the prover.
        
        Args:
            error_message: The error message from the prover
            
        Returns:
            Dictionary with structured error information
        """
        return self._process_feedback_impl(error_message)
    
    @abstractmethod
    def _process_feedback_impl(self, error_message: str) -> Dict[str, Any]:
        """
        Implementation-specific error processing.
        
        Args:
            error_message: The error message from the prover
            
        Returns:
            Dictionary with structured error information
        """
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this prover's scripts.
        
        Returns:
            The file extension (e.g., ".v" for Coq, ".lean" for Lean)
        """
        pass


# Backend registry to manage available provers
class BackendRegistry:
    """Registry of available theorem prover backends."""
    
    _backends = {}
    
    @classmethod
    def register(cls, name: str, backend_class):
        """
        Register a backend class with a name.
        
        Args:
            name: The name to register the backend under
            backend_class: The backend class
        """
        cls._backends[name.lower()] = backend_class
    
    @classmethod
    def get_backend(cls, name: str) -> ProverBackend:
        """
        Get a backend instance by name.
        
        Args:
            name: The name of the backend
            
        Returns:
            An instance of the requested backend
            
        Raises:
            ValueError: If the backend is not registered
        """
        name = name.lower()
        if name not in cls._backends:
            registered = ", ".join(cls._backends.keys())
            raise ValueError(f"Unknown backend '{name}'. Registered backends: {registered}")
        
        return cls._backends[name]()
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """
        List all registered backends.
        
        Returns:
            List of backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def is_backend_available(cls, name: str) -> bool:
        """
        Check if a backend is registered and available.
        
        Args:
            name: The name of the backend
            
        Returns:
            True if the backend is registered and available, False otherwise
        """
        try:
            backend = cls.get_backend(name)
            return backend.is_installed()
        except Exception:
            return False