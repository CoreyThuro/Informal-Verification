# NaturalProofs Integration

This module provides integration with NaturalProofs models for enhanced mathematical understanding and proof processing.

## Overview

NaturalProofs is a system for natural language processing of mathematical content. This integration provides a simplified interface to NaturalProofs functionality while maintaining compatibility with the rest of our proof translation system.

## Components

The integration consists of several key components:

### 1. NaturalProofsInterface

A simplified interface that wraps the core NaturalProofs models and provides methods for:
- Parsing theorems
- Parsing proofs
- Finding similar theorems
- Extracting variables and structure
- Detecting mathematical domains and proof patterns

The interface is designed to gracefully handle cases where NaturalProofs models are not available, falling back to simpler rule-based approaches.

### 2. MathematicalParser

A parser that uses NaturalProofs models to extract and structure mathematical proofs. It provides methods for:
- Parsing proofs with theorem context
- Converting parsed information to our Intermediate Representation (IR)
- Combined parsing and conversion in a single step

### 3. Model Adapters

Simplified adapters for NaturalProofs models:
- `MathematicalModel` for theorem embedding and similarity computation
- `MathTokenizer` for tokenizing mathematical text

## Usage

Here's how to use the NaturalProofs integration:

```python
# Get the NaturalProofs interface
from core import get_naturalproofs_interface

# Initialize the interface (optionally with a path to pre-trained models)
np_interface = get_naturalproofs_interface(model_path="path/to/models")

# Parse a theorem
theorem_info = np_interface.parse_theorem("For any natural number n, n + n is even.")

# Parse a proof with theorem context
proof_info = np_interface.parse_proof(
    "For any natural number n, n + n is even.",
    "Let n be a natural number. Then n + n = 2*n, which is even by definition."
)

# Find similar theorems
similar_theorems = np_interface.find_similar_theorems(
    "For any natural number n, n + n is even.",
    ["For all integers n, n^2 - n is even.", "If n is odd, then n^2 is odd."]
)

# Alternatively, use the MathematicalParser
from core.understanding import MathematicalParser

# Initialize the parser
parser = MathematicalParser(model_path="path/to/models")

# Parse a proof and convert to IR in one step
proof_ir = parser.parse_and_convert(
    "For any natural number n, n + n is even.",
    "Let n be a natural number. Then n + n = 2*n, which is even by definition."
)

# Use the IR for translation
formal_proof = backend.translate(proof_ir)
```

## Integration with the Translation System

The NaturalProofs components are integrated with the rest of the system through the `HybridTranslator`:

```python
from translation.hybrid_translator import HybridTranslator

# Initialize the translator
translator = HybridTranslator(target_prover="coq", model_path="path/to/models")

# Translate a theorem and proof
result = translator.translate(
    "For any natural number n, n + n is even.",
    "Let n be a natural number. Then n + n = 2*n, which is even by definition."
)

# Get the formal proof
formal_proof = result["formal_proof"]
```

The `HybridTranslator` will attempt to use the NaturalProofs parser if available, but will fall back to simpler parsing methods if not.

## Fallback Behavior

The integration is designed to be robust to missing components. If NaturalProofs models are not available:

1. The `NaturalProofsInterface` will use simpler rule-based methods for parsing and analysis
2. The `MathematicalParser` will fall back to basic patterns for structure and domain detection
3. The `HybridTranslator` will use the standard parser instead

This ensures that the system remains functional even without the enhanced capabilities of NaturalProofs.

## Model Configuration

The integration supports several configuration options:

- `model_path`: Path to pre-trained NaturalProofs models
- `use_gpu`: Whether to use GPU acceleration for inference
- `use_llm`: Whether to use language model assistance for translation refinement

## Dependencies

This integration has the following dependencies:

- PyTorch (for model inference)
- Transformers (for tokenization and embedding)
- Our core IR system (for representing proofs)
- Our backend interface (for translating to formal languages)

## Extending the Integration

To extend the NaturalProofs integration:

1. Add new model types to the `MathematicalModel` class
2. Enhance the domain and pattern detection algorithms
3. Improve the fallback methods for more robust behavior when models are unavailable
4. Add additional parsing capabilities for specific mathematical domains