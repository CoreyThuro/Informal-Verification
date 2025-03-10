document.addEventListener('DOMContentLoaded', function() {
    // Copy button functionality
    const copyButton = document.getElementById('copyButton');
    if (copyButton) {
        copyButton.addEventListener('click', function() {
            const formalProof = document.getElementById('formalProof');
            const range = document.createRange();
            range.selectNode(formalProof);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            
            try {
                const successful = document.execCommand('copy');
                const msg = successful ? 'Copied!' : 'Failed to copy!';
                copyButton.textContent = msg;
                
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                }, 2000);
            } catch (err) {
                console.error('Unable to copy', err);
                copyButton.textContent = 'Error!';
                
                setTimeout(function() {
                    copyButton.textContent = 'Copy';
                }, 2000);
            }
            
            window.getSelection().removeAllRanges();
        });
    }
    
    // Clear button functionality
    const clearButton = document.getElementById('clearButton');
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            document.getElementById('proofText').value = '';
        });
    }
    
    // Form validation
    const proofForm = document.getElementById('proofForm');
    if (proofForm) {
        proofForm.addEventListener('submit', function(event) {
            const proofText = document.getElementById('proofText').value.trim();
            
            if (!proofText) {
                event.preventDefault();
                alert('Please enter a proof before submitting.');
            }
        });
    }
});
