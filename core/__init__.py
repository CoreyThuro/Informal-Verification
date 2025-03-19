"""
Core package with NaturalProofs integration.
"""

from core.naturalproofs_integration import get_naturalproofs_interface, NaturalProofsInterface
from core.understanding.mathematical_parser import MathematicalParser

__all__ = [
    'get_naturalproofs_interface',
    'NaturalProofsInterface',
    'MathematicalParser'
]