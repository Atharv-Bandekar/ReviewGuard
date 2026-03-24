// popup.js - E-commerce Only (4-Quadrant Labels)

document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Select Elements
    const analyzeBtn = document.getElementById('analyzeBtn');
    const reviewInput = document.getElementById('reviewInput');
    const resultContainer = document.getElementById('resultContainer');
    const labelBadge = document.getElementById('labelBadge');
    const explainBtn = document.getElementById('explainBtn');
    const explanationBox = document.getElementById('explanationBox');
    const explanationText = document.getElementById('explanationText');

    // Store state
    let currentResult = null;

    // 2. ANALYZE FUNCTION
    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        if (!text) {
            // Shake animation for error
            reviewInput.style.borderColor = "#e53935";
            setTimeout(() => reviewInput.style.borderColor = "#ddd", 500);
            return;
        }

        // Single endpoint for E-commerce reviews
        const endpoint = 'http://127.0.0.1:8000/predict'; 

        // UI Updates: Show loading
        analyzeBtn.innerHTML = '<span class="loading-spinner">↻</span> Scanning...';
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');
        explainBtn.classList.add('hidden');
        explanationBox.classList.add('hidden');

        try {
            // --- Call Backend ---
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            currentResult = data; // Save for XAI

            // Show Result container
            resultContainer.classList.remove('hidden');
            
            // --- Format Badge & Label (4-Quadrant System) ---
            const pct = Math.round(data.confidence * 100);
            const label = data.label;
            
            // Reset styles
            labelBadge.className = "badge"; 
            labelBadge.style.color = '#fff'; // Default text color

            if (label.includes('Genuine-style, Human-written')) {
                labelBadge.style.backgroundColor = '#2e7d32'; // Green
                labelBadge.textContent = `✅ ${label} (${pct}%)`;
            } else if (label.includes('Genuine-style, AI-assisted')) {
                labelBadge.style.backgroundColor = '#fbc02d'; // Yellow
                labelBadge.style.color = '#000'; // Dark text for yellow bg
                labelBadge.textContent = `⚠️ ${label} (${pct}%)`;
            } else if (label.includes('Promotional-style, Human-written')) {
                labelBadge.style.backgroundColor = '#d32f2f'; // Red
                labelBadge.textContent = `🚫 ${label} (${pct}%)`;
            } else if (label.includes('Promotional-style, AI-assisted')) {
                labelBadge.style.backgroundColor = '#b71c1c'; // Dark Red
                labelBadge.textContent = `🚫 ${label} (${pct}%)`;
            } else {
                labelBadge.style.backgroundColor = '#9e9e9e'; // Grey
                labelBadge.textContent = `⚖️ ${label} (${pct}%)`;
            }

            // Show "Why?" button if successful
            explainBtn.classList.remove('hidden');
            explainBtn.textContent = "💡 Why?";
            explainBtn.disabled = false;

        } catch (error) {
            console.error(error);
            labelBadge.textContent = "Connection Error";
            labelBadge.style.backgroundColor = "#9e9e9e";
            resultContainer.classList.remove('hidden');
        } finally {
            analyzeBtn.textContent = "🔍 Analyze Text";
            analyzeBtn.disabled = false;
        }
    });

    // 3. EXPLAIN FUNCTION (STREAMING WITH TYPING EFFECT)
    explainBtn.addEventListener('click', async () => {
        if (!currentResult) return;
        
        // UI Reset
        explainBtn.disabled = true;
        explanationBox.classList.remove('hidden');
        explanationText.innerHTML = "<strong>Analyzing...</strong> "; 
        
        try {
            // Construct URL for GET request
            const params = new URLSearchParams({
                text: reviewInput.value.trim(),
                label: currentResult.label,
                confidence: currentResult.confidence
            });

            const response = await fetch(`http://127.0.0.1:8000/explain_stream?${params.toString()}`);
            
            if (!response.body) {
                throw new Error("ReadableStream not supported.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // Clear "Analyzing..." text
            explanationText.innerHTML = "<strong>AI Insight:</strong> "; 

            // Read the stream loop
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                // Decode chunk
                const chunk = decoder.decode(value, { stream: true });
                
                // Use the smooth typer helper
                await typeOutChunk(explanationText, chunk);
            }

        } catch (error) {
            console.error(error);
            explanationText.innerHTML += "<br>[Connection interrupted]";
        } finally {
            explainBtn.textContent = "💡 Regenerate";
            explainBtn.disabled = false;
        }
    });

    // --- HELPER FUNCTION ---
    async function typeOutChunk(element, text) {
        for (const char of text) {
            // Append safely to innerHTML so we don't destroy the <strong> tag
            element.innerHTML += char;
            // 5ms delay per character = smooth typing effect
            await new Promise(r => setTimeout(r, 5)); 
            
            // Auto-scroll to bottom of the box
            if(element.parentElement) {
                element.parentElement.scrollTop = element.parentElement.scrollHeight;
            }
        }
    }
});