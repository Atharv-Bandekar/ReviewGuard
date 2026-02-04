document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Select Elements
    const analyzeBtn = document.getElementById('analyzeBtn');
    const reviewInput = document.getElementById('reviewInput');
    const resultContainer = document.getElementById('resultContainer');
    const labelBadge = document.getElementById('labelBadge');
    const explainBtn = document.getElementById('explainBtn');
    const explanationBox = document.getElementById('explanationBox');
    const explanationText = document.getElementById('explanationText');
    const modeRadios = document.getElementsByName('mode'); // Get radio buttons

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

        // --- A. Determine Selected Mode ---
        let selectedMode = 'review'; // Default
        for (const radio of modeRadios) {
            if (radio.checked) {
                selectedMode = radio.value;
                break;
            }
        }

        // --- B. Choose Endpoint ---
        // Review Mode -> DeBERTa Model
        // Social Mode -> TinyBERT Model
        const endpoint = selectedMode === 'review' 
            ? 'http://127.0.0.1:8000/predict' 
            : 'http://127.0.0.1:8000/predict_comment';

        // UI Updates: Show loading
        analyzeBtn.innerHTML = '<span class="loading-spinner">↻</span> Scanning...';
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');
        explainBtn.classList.add('hidden');
        explanationBox.classList.add('hidden');

        try {
            // --- C. Call Backend ---
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            currentResult = data; // Save for XAI

            // Show Result container
            resultContainer.classList.remove('hidden');
            
            // --- D. Format Badge & Label ---
            const pct = Math.round(data.confidence * 100);
            labelBadge.className = "badge"; // Reset classes

            if (selectedMode === 'review') {
                // Amazon/Product Logic
                labelBadge.textContent = `${data.label} (${pct}%)`;
                
                if (data.label === 'FAKE') labelBadge.classList.add('fake');
                else if (data.label === 'GENUINE') labelBadge.classList.add('genuine');
                else labelBadge.classList.add('uncertain');
            } 
            else {
                // Social Media Logic
                // Convert "BOT" -> "AI" for display (if backend sends 'BOT')
                const displayLabel = (data.label === 'BOT' || data.label === 'AI') ? 'AI' : data.label;
                labelBadge.textContent = `${displayLabel} (${pct}%)`;

                // 🔴 FIX: Check for both 'BOT' AND 'AI' to apply Red Color
                if (data.label === 'BOT' || data.label === 'AI') {
                    labelBadge.classList.add('bot'); // Uses red style
                }
                else if (data.label === 'HUMAN') {
                    labelBadge.classList.add('human'); // Uses green style
                }
                else {
                    labelBadge.classList.add('uncertain');
                }
            }

            // Show "Why?" button if successful
            explainBtn.classList.remove('hidden');
            explainBtn.textContent = "💡 Why?";
            explainBtn.disabled = false;

        } catch (error) {
            console.error(error);
            labelBadge.textContent = "Connection Error";
            labelBadge.className = "badge uncertain";
            resultContainer.classList.remove('hidden');
        } finally {
            analyzeBtn.textContent = "🔍 Analyze Text";
            analyzeBtn.disabled = false;
        }
    });

    // 3. EXPLAIN FUNCTION (XAI)
    explainBtn.addEventListener('click', async () => {
        if (!currentResult) return;

        // UI Updates
        explainBtn.textContent = "Thinking...";
        explainBtn.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:8000/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: reviewInput.value.trim(),
                    label: currentResult.label,
                    confidence: currentResult.confidence
                })
            });

            const data = await response.json();

            // Show Explanation
            explanationBox.classList.remove('hidden');
            explanationText.innerHTML = `<strong>AI Insight:</strong> ${data.explanation || "No explanation available."}`;

        } catch (error) {
            console.error(error);
            explanationText.textContent = "Could not fetch explanation.";
            explanationBox.classList.remove('hidden');
        } finally {
            explainBtn.textContent = "💡 Why?";
            explainBtn.disabled = false;
        }
    });
});