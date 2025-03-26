# Informal Verification (Minimum Viable Version)

Informal Verification is a tool that automatically translates informal mathematical proofs into formal Coq proofs. It recognizes different proof patterns, detects the mathematical domain, and generates corresponding formal proofs.

## Features

- Recognizes common proof patterns: evenness, induction, contradiction, case analysis, and direct proofs
- Automatically detects mathematical domains based on proof content
- Generates Coq formal proofs with appropriate imports and tactics
- Interactive web interface for ease of use
- Command-line interface for batch processing
- Verification of generated proofs
- Step-by-step translation view
- Error feedback and automatic correction of common issues

## Installation

### Prerequisites

- Python 3.8 or higher
- Coq (optional, for proof verification)

### Setup

1. Clone the repository:
   ```
   git clone [https://github.com/CoreyThuro/Informal-Verification.git]
   cd Informal-Verification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

The web interface provides an easy-to-use environment for translating proofs, with interactive features.

1. Start the web server:
   ```
   python cli.py web
   ```
   
   You can specify host and port:
   ```
   python cli.py web --host 0.0.0.0 --port 8080
   ```

2. Open your browser and navigate to `http://localhost:8000` (or your specified host/port)

3. Enter your theorem and proof in the respective text areas

4. Choose one of the following actions:
   - **Translate**: Convert the proof to Coq
   - **Analyze**: Examine the proof pattern and domain without full translation
   - **Step-by-Step Translation**: See detailed translation process

5. You can also:
   - Verify the generated proof
   - Apply feedback to fix errors
   - Copy the proof to clipboard
   - Load example proofs from different patterns

### Command Line Interface

The CLI is useful for batch processing or integration with other tools.

1. Translate a proof from the command line:
   ```
   python cli.py translate --theorem "For all natural numbers n, n + n is even." --proof "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
   ```

2. Translate from a file:
   ```
   python cli.py translate --file tests/examples/evenness.txt
   ```
   
   The file should have the theorem and proof separated by "Proof:" marker.

3. Save output to a file:
   ```
   python cli.py translate --file tests/examples/induction.txt --output result.v
   ```

4. Verify the generated proof:
   ```
   python cli.py translate --file tests/examples/evenness.txt --verify
   ```

## Project Structure

- `translator.py`: Main translator module
- `patterns/`: Proof pattern recognition and translation
- `knowledge/`: Domain knowledge base
- `coq/`: Coq verification and feedback processing
- `web/`: Web interface related files
- `tests/`: Test cases and examples
- `cli.py`: Command-line interface

## Supported Proof Patterns

1. **Evenness Proofs**: Proofs about numbers being even/divisible by 2
2. **Induction Proofs**: Mathematical induction (base case, inductive step)
3. **Contradiction Proofs**: Assuming the opposite and deriving a contradiction
4. **Case Analysis Proofs**: Splitting into cases (e.g., even vs. odd)
5. **Direct Proofs**: Straightforward logical deduction

## Mathematical Domains

The system detects and specializes translation for various mathematical domains:

- **Number Theory (11)**: Primes, divisibility, etc.
- **Algebra (12-20)**: Groups, rings, fields, etc.
- **Analysis (26-42)**: Limits, continuity, etc.
- **Topology (54-55)**: Open/closed sets, etc.

## Examples

Example proofs are available in the `tests/examples/` directory:
- `evenness.txt`: Proving that n + n is even
- `induction.txt`: Sum of first n natural numbers
- `contradiction.txt`: Irrationality of √2
- `cases.txt`: Proving n² - n is even via case analysis

## Next Steps
Building on the current implementation, here are key areas for enhancement:

Enhanced Domain Knowledge

Expand the MSC (Mathematics Subject Classification) categories database
Add more domain-specific libraries and tactics
Implement deeper semantic understanding of mathematical concepts


Improved Pattern Recognition

Support for hybrid proof patterns (e.g., induction + contradiction)
Enhanced variable and expression extraction
Integrate natural language processing for better theorem statement parsing


Translation Refinement

Implement domain-specific translation strategies as outlined in Phase 3
Support for more complex theorem statements with quantifiers
Generate more elegant and concise proofs with advanced tactics


Verification & Feedback Loop

Complete the verification-driven refinement system
Implement learning from successful translations
Add support for user-guided corrections and preferences


Additional Theorem Provers

Extend beyond Coq to support Lean, Isabelle/HOL, and other formal systems
Create a unified intermediate representation as mentioned in Phase 1
Implement backend-specific optimizations


User Experience Improvements

Create a proof editor with syntax highlighting and autocompletion
Add interactive tutorial mode for learning formal proof structures
Implement collaborative features for team-based proof development


Integration Capabilities

Add API endpoints for integration with other mathematical tools
Support for batch processing of textbook proofs
Create plugins for LaTeX editors and mathematical notebooks

## Contributing

Contributions are welcome! You can extend the system by:
- Adding new proof patterns in `patterns/recognizer.py`
- Enhancing domain knowledge in `knowledge/data/`
- Creating specialized translators in `patterns/translators/`

## License

[MIT License](LICENSE)
