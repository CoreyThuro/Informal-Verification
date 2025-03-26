// Main JavaScript for Proof Translator UI

// Example proofs
const examples = {
    'evenness': {
        theorem: 'For all natural numbers n, n + n is even.',
        proof: 'Let n be any natural number. Then n + n = 2 * n, which is even by definition since it is divisible by 2. This completes the proof.'
    },
    'induction': {
        theorem: 'For all natural numbers n, the sum 0 + 1 + 2 + ... + n equals n(n+1)/2.',
        proof: 'We prove this by induction on n.\n\nBase case: When n = 0, the sum is 0, and the formula gives 0(0+1)/2 = 0, so the formula holds.\n\nInductive step: Assume the formula holds for some k, so 0 + 1 + ... + k = k(k+1)/2.\n\nWe need to show it holds for n = k+1.\n\nWe have:\n0 + 1 + ... + k + (k+1) = k(k+1)/2 + (k+1)\n                          = (k+1)(k/2 + 1)\n                          = (k+1)(k+2)/2\n\nThis is the formula for n = k+1. Thus, by induction, the formula holds for all natural numbers n.'
    },
    'contradiction': {
        theorem: 'There is no rational number r such that r² = 2.',
        proof: 'Suppose, by contradiction, that there exists a rational number r such that r² = 2.\n\nThen r can be written as a/b, where a and b are integers with no common factors (i.e., in lowest form).\n\nSince r² = 2, we have (a/b)² = 2.\n\nThis gives us a² = 2b².\n\nSince a² = 2b², we know that a² is even.\n\nIf a² is even, then a must be even as well (since the square of an odd number is odd).\n\nSo a = 2c for some integer c.\n\nSubstituting this back, we get (2c)² = 2b².\n\nSimplifying: 4c² = 2b².\n\nDividing both sides by 2: 2c² = b².\n\nThis means b² is even, which implies b is even.\n\nBut now we have both a and b are even, which contradicts our assumption that they have no common factors.\n\nTherefore, our initial assumption must be false, and there is no rational number r such that r² = 2.'
    },
    'cases': {
        theorem: 'For any integer n, n² - n is even.',
        proof: 'We will prove this by considering two cases: when n is even and when n is odd.\n\nCase 1: n is even.\nIf n is even, then n = 2k for some integer k.\nSubstituting, we get:\nn² - n = (2k)² - 2k\n       = 4k² - 2k\n       = 2(2k² - k)\n\nSince 2k² - k is an integer, n² - n = 2(2k² - k) is even.\n\nCase 2: n is odd.\nIf n is odd, then n = 2k + 1 for some integer k.\nSubstituting, we get:\nn² - n = (2k + 1)² - (2k + 1)\n       = 4k² + 4k + 1 - 2k - 1\n       = 4k² + 2k\n       = 2(2k² + k)\n\nSince 2k² + k is an integer, n² - n = 2(2k² + k) is even.\n\nSince n² - n is even in both cases, and any integer must be either even or odd, we conclude that for any integer n, n² - n is even.'
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Initialize CodeMirror editors
    const formalProofEditor = CodeMirror(document.getElementById('formal-proof-editor'), {
        mode: 'coq',
        theme: 'default',
        lineNumbers: true,
        lineWrapping: true,
        tabSize: 2,
        indentWithTabs: false,
        extraKeys: {"Tab": (cm) => cm.replaceSelection("  ")}
    });

    const stepCodeEditor = CodeMirror(document.getElementById('step-code-editor'), {
        mode: 'coq',
        theme: 'default',
        lineNumbers: true,
        lineWrapping: true,
        tabSize: 2,
        indentWithTabs: false,
        readOnly: true
    });

    // UI elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const translateBtn = document.getElementById('translate-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const stepTranslateBtn = document.getElementById('step-translate-btn');
    const verifyBtn = document.getElementById('verify-btn');
    const applyFeedbackBtn = document.getElementById('apply-feedback-btn');
    const copyBtn = document.getElementById('copy-btn');
    const prevStepBtn = document.getElementById('prev-step');
    const nextStepBtn = document.getElementById('next-step');
    const stepCounter = document.getElementById('step-counter');
    const stepName = document.getElementById('step-name');
    const stepDescription = document.getElementById('step-description');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');

    // Step-by-step navigation state
    let currentStep = 0;
    let steps = [];

    // Switch tabs
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button and corresponding pane
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Translation action
    translateBtn.addEventListener('click', async () => {
        const theoremText = document.getElementById('theorem').value.trim();
        const proofText = document.getElementById('proof').value.trim();
        
        if (!theoremText || !proofText) {
            alert('Please enter both the theorem and proof.');
            return;
        }
        
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    theorem: theoremText,
                    proof: proofText
                })
            });
            
            const result = await response.json();
            
            // Update UI with results
            document.getElementById('pattern-value').textContent = result.pattern;
            document.getElementById('domain-value').textContent = result.domain;
            
            // Update verification status
            const verificationElement = document.getElementById('verification-value');
            verificationElement.textContent = result.verified ? 'Success' : 'Failed';
            verificationElement.className = result.verified ? 'success' : 'error';
            
            // Set formal proof editor content
            formalProofEditor.setValue(result.formal_proof);
            
            // Show error message if verification failed
            if (!result.verified && result.error_message) {
                errorMessage.textContent = result.error_message;
                errorSection.classList.add('visible');
            } else {
                errorSection.classList.remove('visible');
            }
            
            // Switch to formal proof tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.querySelector('[data-tab="formal-proof"]').classList.add('active');
            document.getElementById('formal-proof').classList.add('active');
            
        } catch (error) {
            console.error('Translation error:', error);
            alert('An error occurred during translation.');
        }
    });

    // Analysis action
    analyzeBtn.addEventListener('click', async () => {
        const theoremText = document.getElementById('theorem').value.trim();
        const proofText = document.getElementById('proof').value.trim();
        
        if (!theoremText || !proofText) {
            alert('Please enter both the theorem and proof.');
            return;
        }
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    theorem: theoremText,
                    proof: proofText
                })
            });
            
            const result = await response.json();
            
            // Update UI with results
            document.getElementById('pattern-value').textContent = result.pattern;
            document.getElementById('domain-value').textContent = result.domain;
            
            // Update pattern info
            const patternInfoElement = document.getElementById('pattern-info');
            if (result.pattern_details && result.pattern_details.description) {
                patternInfoElement.innerHTML = `
                    <p><strong>Pattern:</strong> ${result.pattern}</p>
                    <p><strong>Description:</strong> ${result.pattern_details.description}</p>
                    ${result.pattern_details.examples ? `
                        <p><strong>Examples:</strong></p>
                        <ul>
                            ${result.pattern_details.examples.map(ex => `<li>${ex}</li>`).join('')}
                        </ul>
                    ` : ''}
                `;
            } else {
                patternInfoElement.innerHTML = `<p>No detailed information available for pattern: ${result.pattern}</p>`;
            }
            
            // Update domain info
            const domainInfoElement = document.getElementById('domain-info');
            if (result.domain_info && result.domain_info.name) {
                domainInfoElement.innerHTML = `
                    <p><strong>Domain:</strong> ${result.domain} - ${result.domain_info.name}</p>
                    ${result.domain_info.concepts ? `
                        <p><strong>Key Concepts:</strong> ${result.domain_info.concepts.join(', ')}</p>
                    ` : ''}
                    <p><strong>Required Imports:</strong></p>
                    <ul>
                        ${result.imports.map(imp => `<li><code>${imp}</code></li>`).join('')}
                    </ul>
                `;
            } else {
                domainInfoElement.innerHTML = `<p>No detailed information available for domain: ${result.domain}</p>`;
            }
            
            // Update tactics info
            const tacticsInfoElement = document.getElementById('tactics-info');
            let tacticsHTML = '<p><strong>Pattern-specific Tactics:</strong></p>';
            
            if (result.pattern_tactics && result.pattern_tactics.length > 0) {
                tacticsHTML += '<ul>';
                result.pattern_tactics.forEach(tactic => {
                    tacticsHTML += `<li><code>${tactic.name}</code>: ${tactic.description}</li>`;
                });
                tacticsHTML += '</ul>';
            } else {
                tacticsHTML += '<p>No specific tactics for this pattern.</p>';
            }
            
            tacticsHTML += '<p><strong>Domain-specific Tactics:</strong></p>';
            if (result.domain_tactics && result.domain_tactics.length > 0) {
                tacticsHTML += '<ul>';
                result.domain_tactics.forEach(tactic => {
                    tacticsHTML += `<li><code>${tactic.name}</code>: ${tactic.description}</li>`;
                });
                tacticsHTML += '</ul>';
            } else {
                tacticsHTML += '<p>No specific tactics for this domain.</p>';
            }
            
            tacticsInfoElement.innerHTML = tacticsHTML;
            
            // Switch to analysis tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.querySelector('[data-tab="analysis"]').classList.add('active');
            document.getElementById('analysis').classList.add('active');
            
        } catch (error) {
            console.error('Analysis error:', error);
            alert('An error occurred during analysis.');
        }
    });

    // Step-by-step translation action
    stepTranslateBtn.addEventListener('click', async () => {
        const theoremText = document.getElementById('theorem').value.trim();
        const proofText = document.getElementById('proof').value.trim();
        
        if (!theoremText || !proofText) {
            alert('Please enter both the theorem and proof.');
            return;
        }
        
        try {
            const response = await fetch('/step_translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    theorem: theoremText,
                    proof: proofText
                })
            });
            
            const result = await response.json();
            
            // Store steps and reset step navigation
            steps = result.steps;
            currentStep = 0;
            
            // Update UI with results
            document.getElementById('pattern-value').textContent = steps[0].name.includes('pattern') ? steps[0].description.split(':')[1].trim().split(',')[0] : '-';
            document.getElementById('domain-value').textContent = steps[0].name.includes('domain') ? steps[0].description.split(':')[1].trim().split(',')[1].trim() : '-';
            
            // Initialize step view
            updateStepView();
            
            // Switch to step-by-step tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.querySelector('[data-tab="step-by-step"]').classList.add('active');
            document.getElementById('step-by-step').classList.add('active');
            
        } catch (error) {
            console.error('Step translation error:', error);
            alert('An error occurred during step-by-step translation.');
        }
    });

    // Step navigation
    prevStepBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateStepView();
        }
    });
    
    nextStepBtn.addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            updateStepView();
        }
    });
    
    function updateStepView() {
        if (steps.length === 0) return;
        
        // Update step counter
        stepCounter.textContent = `Step ${currentStep + 1}/${steps.length}`;
        
        // Update step info
        stepName.textContent = steps[currentStep].name;
        stepDescription.textContent = steps[currentStep].description;
        
        // Update step code
        stepCodeEditor.setValue(steps[currentStep].code);
        
        // Enable/disable navigation buttons
        prevStepBtn.disabled = currentStep === 0;
        nextStepBtn.disabled = currentStep === steps.length - 1;
    }

    // Verify proof
    verifyBtn.addEventListener('click', async () => {
        const proofScript = formalProofEditor.getValue();
        
        if (!proofScript.trim()) {
            alert('No proof to verify.');
            return;
        }
        
        try {
            const response = await fetch('/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    proof: proofScript
                })
            });
            
            const result = await response.json();
            
            // Update verification status
            const verificationElement = document.getElementById('verification-value');
            verificationElement.textContent = result.verified ? 'Success' : 'Failed';
            verificationElement.className = result.verified ? 'success' : 'error';
            
            // Show error message if verification failed
            if (!result.verified && result.error) {
                errorMessage.textContent = result.error;
                errorSection.classList.add('visible');
            } else {
                errorSection.classList.remove('visible');
            }
            
        } catch (error) {
            console.error('Verification error:', error);
            alert('An error occurred during verification.');
        }
    });

    // Apply feedback
    applyFeedbackBtn.addEventListener('click', async () => {
        const proofScript = formalProofEditor.getValue();
        const error = errorMessage.textContent;
        
        if (!proofScript.trim()) {
            alert('No proof to fix.');
            return;
        }
        
        if (!error.trim()) {
            alert('No error message to process.');
            return;
        }
        
        try {
            const response = await fetch('/apply_feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    proof: proofScript,
                    error: error
                })
            });
            
            const result = await response.json();
            
            // Update proof editor with fixed proof
            formalProofEditor.setValue(result.fixed_proof);
            
            // Hide error section
            errorSection.classList.remove('visible');
            
            // Verify the fixed proof
            verifyBtn.click();
            
        } catch (error) {
            console.error('Feedback application error:', error);
            alert('An error occurred while applying feedback.');
        }
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const proofScript = formalProofEditor.getValue();
        
        if (!proofScript.trim()) {
            alert('No proof to copy.');
            return;
        }
        
        navigator.clipboard.writeText(proofScript)
            .then(() => {
                // Change button text briefly
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 1500);
            })
            .catch(err => {
                console.error('Copy failed:', err);
                alert('Failed to copy to clipboard.');
            });
    });

    // Load example proofs
    document.querySelectorAll('.load-example-btn').forEach(button => {
        button.addEventListener('click', () => {
            const exampleType = button.parentElement.getAttribute('data-example');
            if (examples[exampleType]) {
                document.getElementById('theorem').value = examples[exampleType].theorem;
                document.getElementById('proof').value = examples[exampleType].proof;
                
                // Scroll to the top of the input section
                const inputSection = document.querySelector('.input-section');
                inputSection.scrollIntoView({ behavior: 'smooth' });
                
                // Highlight the input fields briefly
                const theoremField = document.getElementById('theorem');
                const proofField = document.getElementById('proof');
                
                theoremField.style.transition = 'background-color 1s';
                proofField.style.transition = 'background-color 1s';
                
                theoremField.style.backgroundColor = '#f0f8ff';
                proofField.style.backgroundColor = '#f0f8ff';
                
                setTimeout(() => {
                    theoremField.style.backgroundColor = '';
                    proofField.style.backgroundColor = '';
                }, 1500);
            }
        });
    });
});