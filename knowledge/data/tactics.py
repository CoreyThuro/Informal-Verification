{
  "domains": {
    "11": [
      {"name": "lia", "description": "Linear integer arithmetic solver"},
      {"name": "ring", "description": "Ring simplification"},
      {"name": "omega", "description": "Omega solver for Presburger arithmetic"}
    ],
    "12-20": [
      {"name": "ring", "description": "Ring simplification"},
      {"name": "field", "description": "Field simplification"}
    ],
    "26-42": [
      {"name": "field", "description": "Field simplification"}
    ]
  },
  "patterns": {
    "evenness": [
      {"name": "intro", "args": ["{var}"], "description": "Introduce the variable"},
      {"name": "exists", "args": ["{var}"], "description": "Provide witness for evenness"},
      {"name": "ring", "description": "Solve with ring arithmetic"}
    ],
    "induction": [
      {"name": "induction", "args": ["{var}"], "description": "Induction on variable"},
      {"name": "simpl", "description": "Simplify the goal"},
      {"name": "auto", "description": "Automatic proof search"}
    ],
    "contradiction": [
      {"name": "intro", "args": ["contra"], "description": "Introduce contradiction hypothesis"},
      {"name": "contradiction", "description": "Derive contradiction"}
    ],
    "cases": [
      {"name": "destruct", "args": ["{var}"], "description": "Case analysis on variable"},
      {"name": "simpl", "description": "Simplify each case"},
      {"name": "auto", "description": "Automatic proof search"}
    ]
  }
}