// Main JavaScript for Proof Translator UI

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
});