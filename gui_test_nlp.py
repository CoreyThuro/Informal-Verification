#!/usr/bin/env python3
"""
GUI for testing the NLP-enhanced pattern recognition system.
This allows interactive testing of the system with custom theorem and proof inputs.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
from patterns.enhanced_recognizer import enhanced_recognize_pattern
from patterns.nlp_analyzer import analyze_proof, get_enhanced_pattern
from translator import ProofTranslator

class NLPTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP-Enhanced Proof Analyzer")
        self.root.geometry("900x700")
        
        self.translator = ProofTranslator()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Theorem input
        ttk.Label(input_frame, text="Theorem:").pack(anchor=tk.W)
        self.theorem_text = scrolledtext.ScrolledText(input_frame, height=4)
        self.theorem_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Proof input
        ttk.Label(input_frame, text="Proof:").pack(anchor=tk.W)
        self.proof_text = scrolledtext.ScrolledText(input_frame, height=8)
        self.proof_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Analyze with Enhanced Recognizer", 
                  command=self.analyze_enhanced).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Analyze with NLP Analyzer", 
                  command=self.analyze_nlp).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Translate", 
                  command=self.translate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_results).pack(side=tk.RIGHT, padx=5)
        
        # Example dropdown
        ttk.Label(button_frame, text="Load Example:").pack(side=tk.LEFT, padx=(20, 5))
        self.example_var = tk.StringVar()
        self.examples = self.load_examples()
        example_names = ["Select an example..."] + [f"Example {i+1}: {ex['title']}" for i, ex in enumerate(self.examples)]
        example_dropdown = ttk.Combobox(button_frame, textvariable=self.example_var, values=example_names, width=30)
        example_dropdown.current(0)
        example_dropdown.pack(side=tk.LEFT)
        example_dropdown.bind("<<ComboboxSelected>>", self.load_example)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Load initial examples
        self.load_examples_from_file()
    
    def load_examples(self):
        """Load examples for the dropdown menu."""
        try:
            return self.load_examples_from_file()
        except:
            # Return some default examples if file loading fails
            return [
                {
                    "title": "Simple Evenness",
                    "theorem": "For all natural numbers n, n + n is even.",
                    "proof": "Let n be any natural number. Then n + n = 2 * n, which is even by definition."
                },
                {
                    "title": "Contradiction",
                    "theorem": "The square root of 2 is irrational.",
                    "proof": "Assume, for the sake of contradiction, that the square root of 2 is rational. Then there exist integers p and q with no common factors such that √2 = p/q. Squaring both sides, we get 2 = p²/q², which implies p² = 2q². This means p² is even, which implies p is even. So p = 2k for some integer k. Substituting, we get (2k)² = 2q², or 4k² = 2q², which simplifies to q² = 2k². By the same argument as before, q is even. But this contradicts our assumption that p and q have no common factors. Therefore, √2 cannot be rational."
                }
            ]
    
    def load_examples_from_file(self):
        """Load examples from the semantic examples file."""
        examples = []
        try:
            with open("examples/semantic_examples.txt", 'r') as f:
                content = f.read()
            
            # Split by example markers
            example_sections = content.split("## Example")[1:]
            
            for section in example_sections:
                lines = section.strip().split('\n')
                
                # Extract title
                title = lines[0].strip(": ")
                
                # Find theorem and proof sections
                theorem_start = section.find("### Theorem")
                proof_start = section.find("### Proof")
                next_section = section.find("##", proof_start)
                if next_section == -1:
                    next_section = len(section)
                
                # Extract theorem and proof text
                theorem = section[theorem_start + len("### Theorem"):proof_start].strip()
                proof = section[proof_start + len("### Proof"):next_section].strip()
                
                examples.append({
                    "title": title,
                    "theorem": theorem,
                    "proof": proof
                })
        except Exception as e:
            print(f"Error loading examples: {e}")
        
        return examples
    
    def load_example(self, event):
        """Load the selected example into the input fields."""
        selection = self.example_var.get()
        if selection.startswith("Example"):
            # Extract example number (1-based)
            example_num = int(selection.split(":")[0].replace("Example ", "")) - 1
            if 0 <= example_num < len(self.examples):
                example = self.examples[example_num]
                self.theorem_text.delete(1.0, tk.END)
                self.theorem_text.insert(tk.END, example["theorem"])
                self.proof_text.delete(1.0, tk.END)
                self.proof_text.insert(tk.END, example["proof"])
    
    def analyze_enhanced(self):
        """Analyze using the enhanced recognizer."""
        theorem = self.theorem_text.get(1.0, tk.END).strip()
        proof = self.proof_text.get(1.0, tk.END).strip()
        
        if not theorem or not proof:
            self.show_result("Please enter both theorem and proof.")
            return
        
        pattern, pattern_info = enhanced_recognize_pattern(theorem, proof)
        
        result = f"Enhanced Recognizer Results:\n"
        result += f"Pattern: {pattern}\n"
        result += f"Confidence: {pattern_info['structure_info']['confidence']:.2f}\n\n"
        
        result += "Variables (by importance):\n"
        for var in pattern_info['variables']:
            result += f"  {var}\n"
        
        result += "\nMathematical Entities:\n"
        for entity_type, entities in pattern_info['structure_info']['math_entities'].items():
            if entities:
                result += f"  {entity_type}: {', '.join(entities)}\n"
        
        result += "\nProof Structure:\n"
        for i, step in enumerate(pattern_info['structure_info']['proof_structure'], 1):
            result += f"  Step {i} ({step['type']}): {step['text'][:50]}...\n"
        
        self.show_result(result)
    
    def analyze_nlp(self):
        """Analyze using the NLP analyzer."""
        theorem = self.theorem_text.get(1.0, tk.END).strip()
        proof = self.proof_text.get(1.0, tk.END).strip()
        
        if not theorem or not proof:
            self.show_result("Please enter both theorem and proof.")
            return
        
        nlp_analysis = analyze_proof(theorem, proof)
        
        result = f"NLP Analyzer Results:\n\n"
        
        result += "Pattern Scores:\n"
        for pattern, score in nlp_analysis['pattern_scores'].items():
            result += f"  {pattern}: {score:.2f}\n"
        
        result += "\nMathematical Entities:\n"
        for entity_type, entities in nlp_analysis['entities'].items():
            if entities:
                result += f"  {entity_type}: {', '.join(entities)}\n"
        
        result += "\nVariables:\n"
        for var in nlp_analysis['variables']:
            result += f"  {var}\n"
        
        result += "\nProof Steps:\n"
        for i, step in enumerate(nlp_analysis['steps'], 1):
            result += f"  Step {i} ({step['type']}): {step['text'][:50]}...\n"
        
        self.show_result(result)
    
    def translate(self):
        """Translate the proof using the translator."""
        theorem = self.theorem_text.get(1.0, tk.END).strip()
        proof = self.proof_text.get(1.0, tk.END).strip()
        
        if not theorem or not proof:
            self.show_result("Please enter both theorem and proof.")
            return
        
        result = self.translator.translate(theorem, proof)
        
        output = f"Translation Results:\n\n"
        output += f"Pattern: {result['pattern']}\n"
        output += f"Domain: {result['domain']}\n"
        output += f"Verified: {result['verified']}\n\n"
        output += "Formal Proof:\n"
        output += result['formal_proof']
        
        self.show_result(output)
    
    def show_result(self, text):
        """Display results in the results text area."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
    
    def clear_results(self):
        """Clear the results area."""
        self.results_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = NLPTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
