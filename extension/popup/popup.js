// popup.js - E-commerce Only (4-Quadrant Labels)

document.addEventListener('DOMContentLoaded', () => {

  const analyzeBtn      = document.getElementById('analyzeBtn');
  const reviewInput     = document.getElementById('reviewInput');
  const resultContainer = document.getElementById('resultContainer');
  const labelBadge      = document.getElementById('labelBadge');
  const explainBtn      = document.getElementById('explainBtn');
  const explanationBox  = document.getElementById('explanationBox');
  const explanationText = document.getElementById('explanationText');

  let currentResult = null;

  // --- ENHANCEMENT 2: Render Analytics Dashboard ---
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (!tabs[0] || !tabs[0].url.includes('amazon')) {
          const dashboard = document.getElementById('dashboard');
          if (dashboard) {
              dashboard.innerHTML = '<div style="font-size:12px; color:#666;">Navigate to an Amazon product page to see analytics.</div>';
          }
          return;
      }

      chrome.tabs.sendMessage(tabs[0].id, {action: "GET_STATS"}, function(response) {
          if (response && response.total > 0) {
              document.getElementById('totalScanned').textContent = `(${response.total} scanned)`;
              
              // Update Legend
              document.getElementById('l-gh').textContent = response.genuineHuman;
              document.getElementById('l-ga').textContent = response.genuineAI;
              document.getElementById('l-ph').textContent = response.promoHuman;
              document.getElementById('l-pa').textContent = response.promoAI;

              // Calculate Percentages for the CSS Bar
              const pGH = (response.genuineHuman / response.total) * 100;
              const pGA = (response.genuineAI / response.total) * 100;
              const pPH = (response.promoHuman / response.total) * 100;
              const pPA = (response.promoAI / response.total) * 100;

              // Build the progress bar
              const bar = document.getElementById('statBar');
              if (bar) {
                  bar.innerHTML = `
                      <div class="bar-segment" style="width: ${pGH}%; background: #2e7d32;"></div>
                      <div class="bar-segment" style="width: ${pGA}%; background: #fbc02d;"></div>
                      <div class="bar-segment" style="width: ${pPH}%; background: #d32f2f;"></div>
                      <div class="bar-segment" style="width: ${pPA}%; background: #b71c1c;"></div>
                      <div class="bar-segment" style="width: ${(response.uncertain / response.total) * 100}%; background: #9e9e9e;"></div>
                  `;
              }
          } else {
              const dashboard = document.getElementById('dashboard');
              if (dashboard) {
                  dashboard.innerHTML = '<div style="font-size:12px; color:#666;">Click "Scan Page" on the Amazon page to generate analytics.</div>';
              }
          }
      });
  });

  // --- ANALYZE ---
  analyzeBtn.addEventListener('click', async () => {
    const text = reviewInput.value.trim();
    if (!text) {
      reviewInput.style.borderColor = "#e53935";
      setTimeout(() => reviewInput.style.borderColor = "#ddd", 500);
      return;
    }

    analyzeBtn.innerHTML = '<span class="loading-spinner">↻</span> Scanning...';
    analyzeBtn.disabled  = true;
    resultContainer.classList.add('hidden');
    explainBtn.classList.add('hidden');
    explanationBox.classList.add('hidden');

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
      });

      const data    = await response.json();
      currentResult = data;

      resultContainer.classList.remove('hidden');

      const pct   = Math.round(data.confidence * 100);
      const label = data.label;

      labelBadge.className   = "badge";
      labelBadge.style.color = '#fff';

      // FIX: Decoupled string matching to catch "Possibly AI-assisted"
      if (label.includes('Genuine-style') && label.includes('Human-written')) {
        labelBadge.style.backgroundColor = '#2e7d32';
        labelBadge.textContent = `✅ ${label} (${pct}%)`;
      } else if (label.includes('Genuine-style') && label.includes('AI-assisted')) {
        labelBadge.style.backgroundColor = '#fbc02d';
        labelBadge.style.color = '#000';
        labelBadge.textContent = `⚠️ ${label} (${pct}%)`;
      } else if (label.includes('Promotional-style') && label.includes('Human-written')) {
        labelBadge.style.backgroundColor = '#d32f2f';
        labelBadge.textContent = `🚫 ${label} (${pct}%)`;
      } else if (label.includes('Promotional-style') && label.includes('AI-assisted')) {
        labelBadge.style.backgroundColor = '#b71c1c';
        labelBadge.textContent = `🚫 ${label} (${pct}%)`;
      } else {
        labelBadge.style.backgroundColor = '#9e9e9e';
        labelBadge.textContent = `⚖️ ${label} (${pct}%)`;
      }

      explainBtn.classList.remove('hidden');
      explainBtn.textContent = "💡 Why?";
      explainBtn.disabled    = false;

    } catch (error) {
      console.error(error);
      labelBadge.textContent           = "Connection Error";
      labelBadge.style.backgroundColor = "#9e9e9e";
      resultContainer.classList.remove('hidden');
    } finally {
      analyzeBtn.textContent = "🔍 Analyze Text";
      analyzeBtn.disabled    = false;
    }
  });

  // --- EXPLAIN (STREAMING) ---
  explainBtn.addEventListener('click', async () => {
    if (!currentResult) return;

    explainBtn.disabled = true;
    explanationBox.classList.remove('hidden');

    explanationText.innerHTML = "<strong>AI Insight:</strong> <span class='ai-stream-text'></span>";
    const streamTarget = explanationText.querySelector('.ai-stream-text');

    try {
      const payload = {
        text:       reviewInput.value.trim(),
        label:      currentResult.label,
        confidence: currentResult.confidence,
      };

      const response = await fetch('http://127.0.0.1:8000/explain_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        cache: 'no-store'
      });

      if (!response.body) throw new Error("ReadableStream not supported.");

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        streamTarget.textContent += decoder.decode(value, { stream: true });
        explanationBox.scrollTop = explanationBox.scrollHeight;
      }

    } catch (error) {
      console.error(error);
      streamTarget.textContent += " [Connection interrupted]";
    } finally {
      explainBtn.textContent = "💡 Regenerate";
      explainBtn.disabled    = false;
    }
  });
});