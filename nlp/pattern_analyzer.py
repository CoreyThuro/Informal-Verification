"""
Pattern analyzer for logical proof structures.
Identifies logical flow and patterns in mathematical proofs.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import networkx as nx

# Configure logging
logger = logging.getLogger("pattern_analyzer")

# Try to import spaCy
try:
    import spacy
    HAS_SPACY = True
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        logger.warning("spaCy model 'en_core_web_sm' not found. Pattern analysis may be limited.")
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Pattern analysis may be limited.")
    HAS_SPACY = False
    nlp = None

class LogicalStatement:
    """
    Represents a logical statement in a proof.
    """
    
    def __init__(self, text: str, statement_type: str = "unknown", line_number: int = -1):
        """
        Initialize a logical statement.
        
        Args:
            text: The statement text
            statement_type: The type of statement (assumption, step, conclusion, etc.)
            line_number: The line number in the original proof
        """
        self.text = text
        self.statement_type = statement_type
        self.line_number = line_number
        self.variables = set()
        self.predicates = set()
        self.referenced_statements = []
        self.reasoning_type = "unknown"
        
        # Extract variables and predicates
        self._extract_elements()
    
    def _extract_elements(self) -> None:
        """Extract variables and predicates from the statement."""
        if not HAS_SPACY or nlp is None:
            # Fallback extraction if spaCy is not available
            self._extract_elements_regex()
            return
        
        doc = nlp(self.text)
        
        # Extract variables (single letters, especially if they're nouns)
        for token in doc:
            if len(token.text) == 1 and token.text.isalpha():
                self.variables.add(token.text)
        
        # Extract predicates (verbs and their arguments)
        for token in doc:
            if token.pos_ == "VERB":
                arguments = [child.text for child in token.children]
                predicate = f"{token.lemma_}({', '.join(arguments)})"
                self.predicates.add(predicate)
    
    def _extract_elements_regex(self) -> None:
        """Extract variables and predicates using regex patterns."""
        # Extract variables (single letters)
        variables = re.findall(r'\b([a-zA-Z])\b', self.text)
        self.variables = set(variables)
        
        # Extract predicates (harder without NLP)
        # Simple approximation: Look for function-like patterns
        predicates = re.findall(r'\b([a-zA-Z]+)\s*\(([^)]+)\)', self.text)
        for predicate, args in predicates:
            self.predicates.add(f"{predicate}({args})")
    
    def detect_reasoning_type(self) -> str:
        """
        Detect the type of reasoning used in the statement.
        
        Returns:
            The reasoning type (deduction, induction, etc.)
        """
        # Check for deductive reasoning markers
        deductive_markers = ["therefore", "thus", "hence", "so", "consequently", "it follows that"]
        for marker in deductive_markers:
            if marker in self.text.lower():
                self.reasoning_type = "deduction"
                return self.reasoning_type
        
        # Check for inductive reasoning markers
        inductive_markers = ["induction", "base case", "inductive step", "inductive hypothesis"]
        for marker in inductive_markers:
            if marker in self.text.lower():
                self.reasoning_type = "induction"
                return self.reasoning_type
        
        # Check for case analysis
        case_markers = ["case", "first case", "second case", "if", "when"]
        for marker in case_markers:
            if marker in self.text.lower():
                self.reasoning_type = "case_analysis"
                return self.reasoning_type
        
        # Check for contradiction
        contradiction_markers = ["contradiction", "contrary", "absurd"]
        for marker in contradiction_markers:
            if marker in self.text.lower():
                self.reasoning_type = "contradiction"
                return self.reasoning_type
        
        # Default to unknown
        return self.reasoning_type
    
    def detect_referenced_statements(self, previous_statements: List['LogicalStatement']) -> List[int]:
        """
        Detect references to previous statements.
        
        Args:
            previous_statements: List of previous statements
            
        Returns:
            List of indices of referenced statements
        """
        references = []
        
        # Check for explicit references to previous statements
        reference_markers = ["by", "using", "from", "as in", "according to", "follows from", "given"]
        for marker in reference_markers:
            if marker in self.text.lower():
                # Look for statement numbers or variables from previous statements
                for i, prev_stmt in enumerate(previous_statements):
                    # Check if variables from previous statement are used
                    if len(prev_stmt.variables.intersection(self.variables)) > 0:
                        references.append(i)
                    
                    # Check if the previous statement text is referenced
                    marker_pos = self.text.lower().find(marker)
                    after_marker = self.text[marker_pos + len(marker):].strip()
                    if prev_stmt.text in after_marker:
                        references.append(i)
        
        # Deduplicate
        self.referenced_statements = sorted(set(references))
        return self.referenced_statements
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.statement_type.capitalize()}: {self.text}"
    
    def __repr__(self) -> str:
        """Representation."""
        return f"LogicalStatement({self.statement_type}: {self.text})"

class LogicalGraph:
    """
    Graph representation of the logical structure of a proof.
    """
    
    def __init__(self):
        """Initialize a logical graph."""
        self.statements = []
        self.graph = nx.DiGraph()
    
    def add_statement(self, statement: LogicalStatement) -> int:
        """
        Add a statement to the graph.
        
        Args:
            statement: The logical statement to add
            
        Returns:
            The index of the added statement
        """
        idx = len(self.statements)
        self.statements.append(statement)
        self.graph.add_node(idx, statement=statement)
        return idx
    
    def add_dependency(self, from_idx: int, to_idx: int, dependency_type: str = "depends") -> None:
        """
        Add a dependency between statements.
        
        Args:
            from_idx: Index of the source statement
            to_idx: Index of the target statement
            dependency_type: Type of dependency
        """
        self.graph.add_edge(from_idx, to_idx, type=dependency_type)
    
    def get_roots(self) -> List[int]:
        """
        Get root statements (those with no incoming edges).
        
        Returns:
            List of indices of root statements
        """
        return [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
    
    def get_leaves(self) -> List[int]:
        """
        Get leaf statements (those with no outgoing edges).
        
        Returns:
            List of indices of leaf statements
        """
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
    
    def get_paths(self) -> List[List[int]]:
        """
        Get all paths from roots to leaves.
        
        Returns:
            List of paths (each path is a list of statement indices)
        """
        roots = self.get_roots()
        leaves = self.get_leaves()
        paths = []
        
        for root in roots:
            for leaf in leaves:
                for path in nx.all_simple_paths(self.graph, root, leaf):
                    paths.append(path)
        
        return paths
    
    def analyze_flow(self) -> Dict[str, Any]:
        """
        Analyze the logical flow in the graph.
        
        Returns:
            Dictionary with analysis results
        """
        # Get basic properties
        statement_types = {}
        for i, stmt in enumerate(self.statements):
            if stmt.statement_type not in statement_types:
                statement_types[stmt.statement_type] = []
            statement_types[stmt.statement_type].append(i)
        
        # Find the reasoning chains
        chains = []
        for path in self.get_paths():
            chain = []
            for i in path:
                stmt = self.statements[i]
                chain.append({
                    "index": i,
                    "text": stmt.text,
                    "type": stmt.statement_type,
                    "reasoning": stmt.reasoning_type
                })
            chains.append(chain)
        
        # Analyze the overall structure
        structure = "unknown"
        if "induction" in [stmt.reasoning_type for stmt in self.statements]:
            structure = "induction"
        elif "contradiction" in [stmt.reasoning_type for stmt in self.statements]:
            structure = "contradiction"
        elif "case_analysis" in [stmt.reasoning_type for stmt in self.statements]:
            structure = "case_analysis"
        elif (
            "assumption" in statement_types and 
            "conclusion" in statement_types and 
            len(statement_types.get("step", [])) > 0
        ):
            structure = "direct"
        
        return {
            "structure": structure,
            "statement_types": statement_types,
            "chains": chains,
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "roots": self.get_roots(),
            "leaves": self.get_leaves()
        }
    
    def visualize(self) -> Dict[str, Any]:
        """
        Create a visualization of the graph.
        
        Returns:
            Dictionary with visualization data
        """
        # This would typically create a graph visualization
        # For now, return a simple representation
        nodes = []
        for i, stmt in enumerate(self.statements):
            nodes.append({
                "id": i,
                "label": f"{i}: {stmt.statement_type}",
                "title": stmt.text,
                "type": stmt.statement_type
            })
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "label": data.get("type", "depends")
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }

class PatternAnalyzer:
    """
    Analyzer for logical patterns in mathematical proofs.
    """
    
    def __init__(self, kb=None):
        """
        Initialize the pattern analyzer.
        
        Args:
            kb: Optional knowledge base
        """
        self.kb = kb
        self.pattern_templates = self._load_pattern_templates()
    
    def _load_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load pattern templates.
        
        Returns:
            Dictionary of pattern templates
        """
        if self.kb and hasattr(self.kb, 'patterns'):
            return self.kb.patterns
        else:
            # Default pattern templates
            return {
                "direct": {
                    "description": "Direct proof: starts with assumptions and derives the conclusion directly",
                    "structure": ["assumption", "step", "conclusion"],
                    "keywords": ["assume", "let", "given", "thus", "therefore", "hence", "so"]
                },
                "induction": {
                    "description": "Proof by induction: proves a base case and an inductive step",
                    "structure": ["base_case", "inductive_step", "conclusion"],
                    "keywords": ["induction", "base case", "inductive", "hypothesis", "step", "k", "k+1"]
                },
                "contradiction": {
                    "description": "Proof by contradiction: assumes the negation of the conclusion and derives a contradiction",
                    "structure": ["negation", "steps", "contradiction"],
                    "keywords": ["contradiction", "contrary", "false", "absurd", "suppose not", "assume not"]
                },
                "case_analysis": {
                    "description": "Proof by cases: splits into different cases and proves each separately",
                    "structure": ["case_1", "case_2", "case_n", "conclusion"],
                    "keywords": ["case", "cases", "first", "second", "either", "or", "otherwise"]
                },
                "existence": {
                    "description": "Existence proof: constructs or identifies an object with the required properties",
                    "structure": ["construction", "verification", "conclusion"],
                    "keywords": ["exists", "construct", "find", "example", "witness", "there is", "there exists"]
                },
                "uniqueness": {
                    "description": "Uniqueness proof: assumes two objects satisfy the conditions and proves they are equal",
                    "structure": ["existence", "uniqueness_step", "conclusion"],
                    "keywords": ["unique", "uniquely", "only", "at most one"]
                },
                "evenness_proof": {
                    "description": "Proof that a number or expression is even",
                    "structure": ["assumption", "expression", "conclusion"],
                    "keywords": ["even", "divisible by 2", "2k", "multiple of 2", "2 *", "form 2k"]
                }
            }
    
    def analyze_pattern(self, proof_text: str, domain: str = "") -> Dict[str, Any]:
        """
        Analyze the logical pattern of a proof.
        
        Args:
            proof_text: The proof text
            domain: Optional domain for context
            
        Returns:
            Dictionary with pattern information
        """
        # Extract logical statements
        logical_statements = self._extract_logical_statements(proof_text)
        
        # Identify logical relations between statements
        logical_relations = self._identify_logical_relations(logical_statements)
        
        # Construct a logical flow graph
        logical_graph = self._construct_logical_graph(logical_statements, logical_relations)
        
        # Match against pattern templates
        pattern_matches = self._match_patterns(logical_graph, domain)
        
        # Select the best matching pattern
        best_match = max(pattern_matches, key=lambda x: x["score"]) if pattern_matches else None
        
        # Special handling for evenness proofs - a common case in simple proofs
        is_evenness_proof = self._is_evenness_proof(proof_text)
        if is_evenness_proof:
            # Check if evenness pattern is already detected or has high score
            evenness_matched = False
            if best_match and best_match["name"] == "evenness_proof":
                evenness_matched = True
            
            if not evenness_matched:
                # Create or update evenness pattern
                evenness_pattern = {
                    "name": "evenness_proof",
                    "description": "Proof that a number or expression is even",
                    "score": 0.9,  # High confidence for specialized pattern
                    "matched_keywords": ["even"],
                    "flow_graph": logical_graph.visualize() if hasattr(logical_graph, "visualize") else None
                }
                
                # Make it the best match if it has higher score
                if not best_match or evenness_pattern["score"] > best_match["score"]:
                    best_match = evenness_pattern
                pattern_matches.append(evenness_pattern)
        
        # Extract substructures from the proof
        substructures = self._extract_substructures(logical_statements, logical_graph, 
                                                  best_match["name"] if best_match else "unknown")
        
        return {
            "primary_pattern": {
                "name": best_match["name"] if best_match else "unknown",
                "description": best_match["description"] if best_match else "No recognized pattern",
                "confidence": best_match["score"] if best_match else 0.0
            },
            "all_matches": pattern_matches,
            "logical_structure": {
                "statements": [stmt.text for stmt in logical_statements],
                "statement_types": [stmt.statement_type for stmt in logical_statements],
                "reasoning_types": [stmt.reasoning_type for stmt in logical_statements],
                "graph": logical_graph.analyze_flow() if hasattr(logical_graph, "analyze_flow") else None
            },
            "substructures": substructures
        }
    
    def _extract_logical_statements(self, proof_text: str) -> List[LogicalStatement]:
        """
        Extract logical statements from the proof text.
        
        Args:
            proof_text: The proof text
            
        Returns:
            List of logical statements
        """
        statements = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(proof_text)
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            # Determine statement type based on keywords
            stmt_type = self._determine_statement_type(sentence)
            
            # Create logical statement
            stmt = LogicalStatement(sentence, stmt_type, i)
            
            # Detect reasoning type
            stmt.detect_reasoning_type()
            
            statements.append(stmt)
        
        # Detect referenced statements
        for i, stmt in enumerate(statements):
            if i > 0:
                stmt.detect_referenced_statements(statements[:i])
        
        return statements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The text to split
            
        Returns:
            List of sentences
        """
        if HAS_SPACY and nlp is not None:
            # Use spaCy for better sentence splitting
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to simple regex splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [sent.strip() for sent in sentences if sent.strip()]
    
    def _determine_statement_type(self, sentence: str) -> str:
        """
        Determine the type of a logical statement.
        
        Args:
            sentence: The sentence text
            
        Returns:
            The statement type
        """
        sentence_lower = sentence.lower()
        
        # Check for assumption
        if any(marker in sentence_lower for marker in 
              ["assume", "let", "suppose", "given", "if", "for any", "for all"]):
            return "assumption"
        
        # Check for conclusion
        if any(marker in sentence_lower for marker in 
              ["therefore", "thus", "hence", "so", "we proved", "we conclude", "q.e.d"]):
            return "conclusion"
        
        # Check for case
        if any(marker in sentence_lower for marker in 
              ["case", "first case", "second case", "when", "if"]):
            return "case"
        
        # Check for induction base
        if "base case" in sentence_lower:
            return "induction_base"
        
        # Check for induction step
        if any(marker in sentence_lower for marker in 
              ["inductive step", "inductive hypothesis", "assume for k"]):
            return "induction_step"
        
        # Check for contradiction
        if any(marker in sentence_lower for marker in 
              ["contradiction", "contrary", "absurd", "impossible"]):
            return "contradiction"
        
        # Default to step
        return "step"
    
    def _identify_logical_relations(self, statements: List[LogicalStatement]) -> List[Tuple[int, int, str]]:
        """
        Identify logical relations between statements.
        
        Args:
            statements: List of logical statements
            
        Returns:
            List of (from_idx, to_idx, relation_type) tuples
        """
        relations = []
        
        # Process each statement starting from the second one
        for i in range(1, len(statements)):
            curr_stmt = statements[i]
            
            # Check for explicit references
            for ref_idx in curr_stmt.referenced_statements:
                if ref_idx < i:
                    relations.append((ref_idx, i, "references"))
            
            # If no explicit references, consider previous statement
            if not curr_stmt.referenced_statements and i > 0:
                # Check for variable overlap
                prev_stmt = statements[i-1]
                if len(prev_stmt.variables.intersection(curr_stmt.variables)) > 0:
                    relations.append((i-1, i, "uses_variables"))
                else:
                    # Default sequential relation
                    relations.append((i-1, i, "follows"))
        
        return relations
    
    def _construct_logical_graph(self, 
                               statements: List[LogicalStatement], 
                               relations: List[Tuple[int, int, str]]) -> LogicalGraph:
        """
        Construct a logical flow graph.
        
        Args:
            statements: List of logical statements
            relations: List of relations between statements
            
        Returns:
            A logical graph
        """
        graph = LogicalGraph()
        
        # Add statements to the graph
        for stmt in statements:
            graph.add_statement(stmt)
        
        # Add relations as edges
        for from_idx, to_idx, rel_type in relations:
            graph.add_dependency(from_idx, to_idx, rel_type)
        
        return graph
    
    def _match_patterns(self, logical_graph: LogicalGraph, domain: str) -> List[Dict[str, Any]]:
        """
        Match the logical graph against pattern templates.
        
        Args:
            logical_graph: The logical graph
            domain: The domain for context
            
        Returns:
            List of matched patterns with scores
        """
        matches = []
        
        # Get statement types
        statement_types = {}
        for i, stmt in enumerate(logical_graph.statements):
            if stmt.statement_type not in statement_types:
                statement_types[stmt.statement_type] = []
            statement_types[stmt.statement_type].append(i)
        
        # Get reasoning types
        reasoning_types = {}
        for i, stmt in enumerate(logical_graph.statements):
            if stmt.reasoning_type not in reasoning_types:
                reasoning_types[stmt.reasoning_type] = []
            reasoning_types[stmt.reasoning_type].append(i)
        
        # Match each pattern template
        for pattern_name, pattern_info in self.pattern_templates.items():
            score = 0.0
            matched_keywords = []
            
            # Check for required statement types
            structure = pattern_info.get("structure", [])
            for required_type in structure:
                if required_type in statement_types:
                    score += 1.0
            
            # Normalize by the number of required types
            if structure:
                score /= len(structure)
            
            # Check for keywords
            keywords = pattern_info.get("keywords", [])
            proof_text = " ".join([stmt.text.lower() for stmt in logical_graph.statements])
            for keyword in keywords:
                if keyword in proof_text:
                    score += 0.2  # Smaller boost for keywords
                    matched_keywords.append(keyword)
            
            # Check reasoning types
            if pattern_name == "induction" and "induction" in reasoning_types:
                score += 0.5
            elif pattern_name == "contradiction" and "contradiction" in reasoning_types:
                score += 0.5
            elif pattern_name == "case_analysis" and "case_analysis" in reasoning_types:
                score += 0.5
            
            # Boost score for domain-specific patterns
            if domain:
                domain_boost = self._get_domain_pattern_boost(pattern_name, domain)
                score *= domain_boost
            
            # Only include if score is significant
            if score > 0.2:
                matches.append({
                    "name": pattern_name,
                    "description": pattern_info.get("description", ""),
                    "score": min(score, 1.0),  # Cap at 1.0
                    "matched_keywords": matched_keywords,
                    "flow_graph": logical_graph.visualize() if hasattr(logical_graph, "visualize") else None
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches
    
    def _get_domain_pattern_boost(self, pattern_name: str, domain: str) -> float:
        """
        Get a domain-specific boost for a pattern.
        
        Args:
            pattern_name: The pattern name
            domain: The domain code
            
        Returns:
            Boost factor (1.0 = no boost)
        """
        # Domain-specific boosts for common patterns
        domain_boosts = {
            "11": {  # Number Theory
                "induction": 1.5,
                "contradiction": 1.3,
                "evenness_proof": 1.5
            },
            "12-15": {  # Algebra
                "direct": 1.2,
                "case_analysis": 1.2
            },
            "26-42": {  # Analysis
                "contradiction": 1.4,
                "existence": 1.3
            },
            "54-55": {  # Topology
                "contradiction": 1.3,
                "existence": 1.2
            }
        }
        
        # Get the boost if available
        if domain in domain_boosts and pattern_name in domain_boosts[domain]:
            return domain_boosts[domain][pattern_name]
        else:
            return 1.0  # No boost
    
    def _extract_substructures(self, 
                              statements: List[LogicalStatement], 
                              logical_graph: LogicalGraph,
                              pattern_name: str) -> Dict[str, str]:
        """
        Extract substructures from the proof.
        
        Args:
            statements: List of logical statements
            logical_graph: The logical graph
            pattern_name: The pattern name
            
        Returns:
            Dictionary mapping substructure names to text
        """
        substructures = {}
        
        # Extract based on statement types
        for i, stmt in enumerate(statements):
            if stmt.statement_type == "assumption":
                substructures["assumption"] = stmt.text
            elif stmt.statement_type == "conclusion":
                substructures["conclusion"] = stmt.text
            elif stmt.statement_type == "induction_base":
                substructures["base_case"] = stmt.text
            elif stmt.statement_type == "induction_step":
                substructures["inductive_step"] = stmt.text
            elif stmt.statement_type == "case" and "case_1" not in substructures:
                substructures["case_1"] = stmt.text
            elif stmt.statement_type == "case" and "case_1" in substructures and "case_2" not in substructures:
                substructures["case_2"] = stmt.text
            elif stmt.statement_type == "contradiction":
                substructures["contradiction"] = stmt.text
        
        # Extract pattern-specific substructures
        if pattern_name == "evenness_proof":
            # Look for the expression being proven even
            for stmt in statements:
                even_matches = re.search(r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+even', stmt.text, re.IGNORECASE)
                if even_matches:
                    substructures["expression"] = even_matches.group(1)
                    break
                
                # Alternative pattern with "divisible by 2"
                divisible_matches = re.search(r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+divisible\s+by\s+2', 
                                             stmt.text, re.IGNORECASE)
                if divisible_matches:
                    substructures["expression"] = divisible_matches.group(1)
                    break
        
        # If no specific substructures found, include representative statements
        if len(substructures) == 0:
            if statements:
                substructures["representative"] = statements[0].text
            if len(statements) > 1:
                substructures["key_step"] = statements[len(statements) // 2].text
            if len(statements) > 2:
                substructures["final"] = statements[-1].text
        
        return substructures
    
    def _is_evenness_proof(self, proof_text: str) -> bool:
        """
        Check if this is a proof about evenness.
        
        Args:
            proof_text: The proof text
            
        Returns:
            True if this is an evenness proof, False otherwise
        """
        # Check for key indicators
        if "even" not in proof_text.lower():
            return False
        
        # Look for patterns indicating evenness
        patterns = [
            r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+even',
            r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+is\s+divisible\s+by\s+2',
            r'([a-zA-Z0-9\+\-\*\/\(\)]+)\s+=\s+2\s*\*\s*[a-zA-Z0-9]+',
            r'([a-zA-Z])\s*\+\s*\1'  # Pattern like x + x
        ]
        
        for pattern in patterns:
            if re.search(pattern, proof_text, re.IGNORECASE):
                return True
        
        return False

def analyze_proof_pattern(proof_text: str, domain: str = "", kb=None) -> Dict[str, Any]:
    """
    Analyze the logical pattern of a proof.
    
    Args:
        proof_text: The proof text
        domain: Optional domain for context
        kb: Optional knowledge base
        
    Returns:
        Dictionary with pattern analysis results
    """
    analyzer = PatternAnalyzer(kb)
    return analyzer.analyze_pattern(proof_text, domain)