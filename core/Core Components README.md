# Core Components

This directory contains the core components for mathematical understanding and NaturalProofs integration.

## Directory Structure

```
core/
├── __init__.py                  # Package exports
├── naturalproofs_integration.py # NaturalProofs interface
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── pairwise_model.py        # Pairwise model for similarity
│   ├── sequence_model.py        # Sequence model for generation
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── utils.py             # Helper functions
├── tokenization/                # Tokenization utilities
│   ├── __init__.py
│   ├── tokenizer.py             # Mathematical tokenizer
│   └── tokenize_pairwise.py     # Tokenization script
└── understanding/               # Understanding components
    ├── __init__.py
    └── mathematical_parser.py   # Mathematical parser
```

## Components Overview

### NaturalProofs Integration

The `naturalproofs_integration.py` provides a unified interface to the NaturalProofs functionality. It handles integration with the rest of the system and provides fallback methods when models are not available.

See the [NaturalProofs Integration README](./NATURALPROOFS_INTEGRATION.md) for detailed information.

### Models

The `models/` directory contains simplified implementations of the NaturalProofs models:

- `pairwise_model.py`: Contains the `MathematicalModel` class for encoding theorems and computing similarity
- `sequence_model.py`: Contains the `SequenceModel` class for sequence generation

### Tokenization

The `tokenization/` directory contains tokenization utilities:

- `tokenizer.py`: Contains the `MathTokenizer` class for tokenizing mathematical text
- `tokenize_pairwise.py`: A script for pairwise tokenization of datasets

### Understanding

The `understanding/` directory contains components for mathematical understanding:

- `mathematical_parser.py`: Contains the `MathematicalParser` class that uses NaturalProofs models to parse and analyze mathematical proofs

## Usage

Here's how to use the core components:

```python
# Import the NaturalProofs interface
from core import get_naturalproofs_interface, MathematicalParser

# Get the interface
np_interface = get_naturalproofs_interface()

# Parse a theorem
theorem_info = np_interface.parse_theorem("For any natural number n, n + n is even.")

# Or use the parser directly
parser = MathematicalParser()
proof_ir = parser.parse_and_convert(
    "For any natural number n, n + n is even.",
    "Let n be a natural number. Then n + n = 2*n, which is even by definition."
)
```

## Integration with the System

The core components are integrated with the rest of the system through:

1. The `HybridTranslator` in `translation/hybrid_translator.py`
2. The IR system in `ir/proof_ir.py`
3. The backend interfaces in `backends/`

## Robustness

The core components are designed to be robust to missing dependencies. If NaturalProofs models or external libraries are not available, the system will fall back to simpler rule-based methods.

## Extension

To extend the core components:

1. Add new model types to the `models/` directory
2. Enhance the tokenization utilities for new token types
3. Improve the parsing capabilities in `understanding/mathematical_parser.py`
4. Update the interface in `naturalproofs_integration.py`