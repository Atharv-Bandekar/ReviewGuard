// content.js - Optimized Batch Processing + Streaming Explanations (E-Commerce Only)

(() => {
  const API_BASE = "http://127.0.0.1:8000";
  const PREDICT_BATCH_URL = `${API_BASE}/predict_batch`; 
  const EXPLAIN_STREAM_URL = `${API_BASE}/explain_stream`;

  // --- 1. SITE DETECTION ---
  const HOST = window.location.hostname;
  if (!HOST.includes('amazon')) {
      console.log("[ReviewGuard] Inactive on this domain. E-commerce only.");
      return; // Exit script if not on Amazon
  }

  console.log(`[ReviewGuard] Active on Amazon`);

  // --- 2. SELECTORS ---
  function getItemsToAnalyze() {
      const nodes = Array.from(document.querySelectorAll('div[id^="customer_review-"]'));
      return nodes.filter(n => n.offsetParent !== null);
  }

  function extractText(node) {
      const standardBox = node.querySelector('[data-hook="review-body"] span') || 
                          node.querySelector('.review-text-content span');
      if (standardBox) return standardBox.innerText.trim();
      
      const clone = node.cloneNode(true);
      ['.a-profile', '.review-date', '.video-block', '.review-title', '.review-comments'].forEach(s => 
        clone.querySelectorAll(s).forEach(n => n.remove())
      );
      return clone.innerText.trim().replace(/Read more|Helpful|Report/gi, '');
  }

  // --- 3. UI HELPERS ---

  async function fetchExplanation(text, label, confidence, container) {
    try {
        container.innerHTML = '<span class="loading-spinner">↻</span> <i>Asking AI...</i>';
        container.style.display = 'block';

        const params = new URLSearchParams({
            text: text.substring(0, 300),
            label: label,
            confidence: confidence
        });

        const response = await fetch(`${EXPLAIN_STREAM_URL}?${params.toString()}`);
        if (!response.body) throw new Error("Stream not supported");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        container.innerHTML = '<b>AI Logic:</b> ';
        
        // Match border color to the 4-quadrant severity
        let borderColor = '#43a047'; // Green
        if (label.includes('Promotional-style, AI-assisted')) borderColor = '#b71c1c'; // Dark Red
        else if (label.includes('Promotional-style')) borderColor = '#e53935'; // Red
        else if (label.includes('AI-assisted')) borderColor = '#fbc02d'; // Yellow
        
        container.style.borderLeft = `3px solid ${borderColor}`;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            await typeOutChunk(container, chunk);
        }

    } catch (e) {
        console.error(e);
        container.innerHTML = 'Error connecting to AI.';
    }
  }

  async function typeOutChunk(element, text) {
    for (const char of text) {
        element.innerHTML += char; 
        await new Promise(r => setTimeout(r, 5)); 
    }
  }

  function createBadge(label, confidence) {
    const span = document.createElement('span');
    span.className = 'rg-badge';
    span.style.cssText = 'display:inline-block; padding:2px 6px; margin:0 5px; border-radius:4px; font-weight:700; font-size:11px; color:white; vertical-align:middle; font-family:sans-serif; z-index:9999;';
    
    const pct = (confidence * 100).toFixed(0);

    // Parse the 4-Quadrant Labels
    if (label.includes('Genuine-style, Human-written')) {
        span.style.backgroundColor = '#2e7d32'; // Green
        span.textContent = `✅ ${label} ${pct}%`;
    } else if (label.includes('Genuine-style, AI-assisted')) {
        span.style.backgroundColor = '#fbc02d'; // Warning Yellow
        span.style.color = '#000'; // Dark text for readability
        span.textContent = `⚠️ ${label} ${pct}%`;
    } else if (label.includes('Promotional-style, Human-written')) {
        span.style.backgroundColor = '#d32f2f'; // Red
        span.textContent = `🚫 ${label} ${pct}%`;
    } else if (label.includes('Promotional-style, AI-assisted')) {
        span.style.backgroundColor = '#b71c1c'; // Darker Red
        span.textContent = `🚫 ${label} ${pct}%`;
    } else {
        span.style.backgroundColor = '#9e9e9e'; // Grey fallback
        span.textContent = `⚖️ ${label} ${pct}%`;
    }
    return span;
  }

  function attachBadge(node, label, confidence, text) {
    if (node.getAttribute('data-rg-status') === 'done') return;
    if (node.querySelector('.rg-badge')) {
        node.setAttribute('data-rg-status', 'done');
        return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'rg-wrapper';
    wrapper.style.cssText = 'display:inline-flex; align-items:center; gap:5px; margin: 5px 0;';

    const badge = createBadge(label, confidence);
    wrapper.appendChild(badge);

    if (label !== 'ERR') {
        const btn = document.createElement('button');
        btn.textContent = '💡 Why?';
        btn.style.cssText = 'border:1px solid #ccc; background:#fff; cursor:pointer; font-size:11px; padding:2px 8px; border-radius:10px; color:#333;';
        
        const explainBox = document.createElement('div');
        explainBox.style.cssText = 'display:none; margin-top:5px; font-size:13px; color:#333; background:#f0f2f5; padding:8px; border-radius:4px; width:100%; line-height: 1.4;';

        btn.onclick = (e) => {
            e.preventDefault();
            btn.style.display = 'none'; 
            fetchExplanation(text, label, confidence, explainBox);
        };
        wrapper.appendChild(btn);
        
        const header = node.querySelector('.a-profile') || node.querySelector('.review-header');
        if (header) {
             header.parentElement.insertBefore(wrapper, header.nextSibling);
             wrapper.insertAdjacentElement('afterend', explainBox);
        } else {
             node.prepend(wrapper);
             wrapper.insertAdjacentElement('afterend', explainBox);
        }
    }
    node.setAttribute('data-rg-status', 'done');
  }

  // --- 4. OPTIMIZED BATCH PROCESSING LOOP ---
  let processing = false;
  
  async function runAnalysis() {
    if (processing) return;
    processing = true;
    updateButton("⏳ Scanning...");

    try { document.querySelectorAll('.a-expander-header a').forEach(btn => btn.click()); } catch(e) {}

    const items = getItemsToAnalyze();
    const newItems = items.filter(i => !i.getAttribute('data-rg-status'));

    if (newItems.length === 0) {
        updateButton("✅ No New Items");
        setTimeout(() => updateButton("🔍 Scan Page"), 2000);
        processing = false;
        return;
    }

    const BATCH_SIZE = 5; 
    const textBuffer = [];
    const nodeBuffer = [];

    newItems.forEach(node => {
        const text = extractText(node);
        if (text && text.length > 5) {
            textBuffer.push(text);
            nodeBuffer.push(node);
            node.setAttribute('data-rg-status', 'pending'); 
        } else {
            node.setAttribute('data-rg-status', 'done'); 
        }
    });

    for (let i = 0; i < textBuffer.length; i += BATCH_SIZE) {
        const texts = textBuffer.slice(i, i + BATCH_SIZE);
        const nodes = nodeBuffer.slice(i, i + BATCH_SIZE);
        
        try {
            const res = await fetch(PREDICT_BATCH_URL, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ texts: texts })
            });
            const data = await res.json();

            if (data.results) {
                data.results.forEach((result, idx) => {
                    attachBadge(nodes[idx], result.label, result.confidence, texts[idx]);
                });
            }
        } catch (e) {
            console.error("Batch Error:", e);
            nodes.forEach(n => n.removeAttribute('data-rg-status'));
        }
    }

    updateButton("🔍 Scan More");
    processing = false;
  }

  // --- 5. FLOATING BUTTON ---
  function updateButton(text) {
    const btn = document.getElementById('rg-float-btn');
    if (btn) btn.textContent = text;
  }

  function injectButton() {
    if (document.getElementById('rg-float-btn')) return;
    const btn = document.createElement('button');
    btn.id = 'rg-float-btn';
    btn.textContent = '🔍 Scan Page';
    btn.style.cssText = `position: fixed; bottom: 20px; right: 20px; z-index: 2147483647; padding: 12px 20px; background: #232f3e; color: #fff; border: 2px solid #fff; border-radius: 30px; font-family: sans-serif; font-weight: bold; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.3); transition: transform 0.2s;`;
    
    btn.onmouseover = () => btn.style.transform = 'scale(1.05)';
    btn.onmouseout = () => btn.style.transform = 'scale(1)';
    btn.onclick = runAnalysis;
    document.body.appendChild(btn);
  }

  injectButton();
  setInterval(injectButton, 1000);

})();