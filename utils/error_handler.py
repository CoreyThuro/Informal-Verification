"""
Error handler for the proof translation system.
Provides centralized error handling and logging.
"""

import sys
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Type, Union
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("proof_translator")

# Error categories
ERROR_CATEGORIES = {
    "parsing": "Error in parsing the input",
    "translation": "Error in translating the proof",
    "verification": "Error in verifying the proof",
    "llm": "Error in language model interaction",
    "io": "Error in file input/output",
    "network": "Network or API error",
    "system": "System error",
    "user": "User input error",
    "unknown": "Unknown error"
}

class TranslationError(Exception):
    """Base exception class for all proof translation errors."""
    
    def __init__(self, message: str, category: str = "unknown", 
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: The error message
            category: The error category
            details: Optional error details
        """
        self.message = message
        self.category = category
        self.details = details or {}
        super().__init__(message)

class ParsingError(TranslationError):
    """Error in parsing the proof."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: The error message
            details: Optional error details
        """
        super().__init__(message, "parsing", details)

class TranslationProcessError(TranslationError):
    """Error in the translation process."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: The error message
            details: Optional error details
        """
        super().__init__(message, "translation", details)

class VerificationError(TranslationError):
    """Error in verifying the proof."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: The error message
            details: Optional error details
        """
        super().__init__(message, "verification", details)

class LLMError(TranslationError):
    """Error in language model interaction."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: The error message
            details: Optional error details
        """
        super().__init__(message, "llm", details)

class ErrorHandler:
    """
    Central error handler for the proof translation system.
    """
    
    def __init__(self, logging_enabled: bool = True, exit_on_error: bool = False):
        """
        Initialize the error handler.
        
        Args:
            logging_enabled: Whether to log errors
            exit_on_error: Whether to exit the program on error
        """
        self.logging_enabled = logging_enabled
        self.exit_on_error = exit_on_error
        self.error_handlers: Dict[str, List[Callable]] = {}
        
        # Register default handlers for each category
        for category in ERROR_CATEGORIES:
            self.error_handlers[category] = []
    
    def register_handler(self, category: str, handler: Callable) -> None:
        """
        Register an error handler for a category.
        
        Args:
            category: The error category
            handler: The handler function
        """
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        
        self.error_handlers[category].append(handler)
    
    def handle_error(self, error: Union[Exception, str], 
                     category: str = "unknown", 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error.
        
        Args:
            error: The error or error message
            category: The error category
            details: Optional error details
        """
        # Convert string to exception if needed
        if isinstance(error, str):
            error = TranslationError(error, category, details)
        
        # Extract error info
        error_info = self._extract_error_info(error, category, details)
        
        # Log the error
        if self.logging_enabled:
            self._log_error(error_info)
        
        # Call registered handlers
        self._call_handlers(error_info)
        
        # Exit if configured to do so
        if self.exit_on_error:
            sys.exit(1)
    
    def _extract_error_info(self, error: Exception, 
                           category: str = "unknown", 
                           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract information from an error.
        
        Args:
            error: The error
            category: The error category
            details: Optional error details
            
        Returns:
            Dictionary with error information
        """
        # Initialize with basic info
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "category": category,
            "details": details or {},
            "traceback": traceback.format_exc()
        }
        
        # If it's our custom error, extract additional info
        if isinstance(error, TranslationError):
            error_info["category"] = error.category
            error_info["details"].update(error.details)
        
        return error_info
    
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """
        Log an error.
        
        Args:
            error_info: The error information
        """
        # Log the error
        category_desc = ERROR_CATEGORIES.get(error_info["category"], "Unknown error")
        
        logger.error(f"{category_desc}: {error_info['message']}")
        
        # Log details for debug level
        if logger.isEnabledFor(logging.DEBUG):
            details_str = json.dumps(error_info["details"], indent=2)
            logger.debug(f"Error details: {details_str}")
            logger.debug(f"Traceback: {error_info['traceback']}")
    
    def _call_handlers(self, error_info: Dict[str, Any]) -> None:
        """
        Call registered handlers for an error.
        
        Args:
            error_info: The error information
        """
        category = error_info["category"]
        
        # Call handlers for this category
        if category in self.error_handlers:
            for handler in self.error_handlers[category]:
                try:
                    handler(error_info)
                except Exception as e:
                    logger.error(f"Error in error handler: {str(e)}")
        
        # Call handlers for all errors
        if "all" in self.error_handlers:
            for handler in self.error_handlers["all"]:
                try:
                    handler(error_info)
                except Exception as e:
                    logger.error(f"Error in error handler: {str(e)}")


# Create a global error handler instance
global_error_handler = ErrorHandler()

# Utility functions for use in other modules

def handle_error(error: Union[Exception, str], 
                category: str = "unknown", 
                details: Optional[Dict[str, Any]] = None) -> None:
    """
    Handle an error using the global error handler.
    
    Args:
        error: The error or error message
        category: The error category
        details: Optional error details
    """
    global_error_handler.handle_error(error, category, details)

def register_error_handler(category: str, handler: Callable) -> None:
    """
    Register an error handler using the global error handler.
    
    Args:
        category: The error category
        handler: The handler function
    """
    global_error_handler.register_handler(category, handler)

def try_except(func: Callable, error_category: str = "unknown", 
              custom_message: Optional[str] = None) -> Callable:
    """
    Decorator to wrap a function in try-except and handle errors.
    
    Args:
        func: The function to wrap
        error_category: The error category
        custom_message: Optional custom error message
        
    Returns:
        The wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            details = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            message = custom_message or str(e)
            
            handle_error(e, error_category, details)
            
            # Re-raise the exception to allow further handling
            raise
    
    return wrapper