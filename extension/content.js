// content.js - Optimized Batch Processing + Streaming Explanations

(() => {
  const API_BASE = "http://127.0.0.1:8000";
  // Endpoints
  const PREDICT_BATCH_URL = `${API_BASE}/predict_batch`; 
  const PREDICT_COMMENT_BATCH_URL = `${API_BASE}/predict_comment_batch`; 
  const EXPLAIN_STREAM_URL = `${API_BASE}/explain_stream`; // <--- NEW ENDPOINT

  // --- 1. SITE DETECTION ---
  const HOST = window.location.hostname;
  let SITE_TYPE = 'UNKNOWN';

  if (HOST.includes('amazon')) SITE_TYPE = 'AMAZON';
  else if (HOST.includes('youtube')) SITE_TYPE = 'YOUTUBE';
  else if (HOST.includes('twitter') || HOST.includes('x.com')) SITE_TYPE = 'TWITTER';

  console.log(`[ReviewGuard] Active on ${SITE_TYPE}`);

  // --- 2. SELECTORS ---
  function getItemsToAnalyze() {
    if (SITE_TYPE === 'AMAZON') {
      const nodes = Array.from(document.querySelectorAll('div[id^="customer_review-"]'));
      return nodes.filter(n => n.offsetParent !== null);
    } 
    else if (SITE_TYPE === 'YOUTUBE') {
      return Array.from(document.querySelectorAll('#content-text'));
    } 
    else if (SITE_TYPE === 'TWITTER') {
      return Array.from(document.querySelectorAll('[data-testid="tweetText"]'));
    }
    return [];
  }

  function extractText(node) {
    if (SITE_TYPE === 'AMAZON') {
      const standardBox = node.querySelector('[data-hook="review-body"] span') || 
                          node.querySelector('.review-text-content span');
      if (standardBox) return standardBox.innerText.trim();
      
      const clone = node.cloneNode(true);
      ['.a-profile', '.review-date', '.video-block', '.review-title', '.review-comments'].forEach(s => 
        clone.querySelectorAll(s).forEach(n => n.remove())
      );
      return clone.innerText.trim().replace(/Read more|Helpful|Report/gi, '');
    } 
    else {
      return node.innerText.trim();
    }
  }

  // --- 3. UI HELPERS ---

  // 🟢 NEW: Streaming Explanation Function
  async function fetchExplanation(text, label, confidence, container) {
    try {
        // UI Reset
        container.innerHTML = '<span class="loading-spinner">↻</span> <i>Asking AI...</i>';
        container.style.display = 'block';

        // Construct URL Params for GET request
        const params = new URLSearchParams({
            text: text.substring(0, 300), // Limit text length for URL safety
            label: label,
            confidence: confidence
        });

        // Fetch Stream
        const response = await fetch(`${EXPLAIN_STREAM_URL}?${params.toString()}`);
        
        if (!response.body) throw new Error("Stream not supported");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // Prepare container for typing
        container.innerHTML = '<b>AI Logic:</b> ';
        container.style.borderLeft = `3px solid ${['FAKE', 'AI', 'BOT'].includes(label) ? '#e53935' : '#43a047'}`;

        // Read Loop
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            
            // Smooth Type Effect
            await typeOutChunk(container, chunk);
        }

    } catch (e) {
        console.error(e);
        container.innerHTML = 'Error connecting to AI.';
    }
  }

  // 🟢 NEW: Smooth Typing Helper
  async function typeOutChunk(element, text) {
    // Append text node to avoid destroying '<b>AI Logic:</b>'
    // or just append to innerHTML if simple
    for (const char of text) {
        element.innerHTML += char; 
        // 5ms delay = smooth typing look
        await new Promise(r => setTimeout(r, 5)); 
    }
  }

  function createBadge(label, confidence, isVerified) {
    const span = document.createElement('span');
    span.className = 'rg-badge';
    span.style.cssText = 'display:inline-block; padding:2px 6px; margin:0 5px; border-radius:4px; font-weight:700; font-size:11px; color:white; vertical-align:middle; font-family:sans-serif; z-index:9999;';
    
    const pct = (confidence * 100).toFixed(0);

    if (label === 'GENUINE') {
        span.style.backgroundColor = '#2e7d32'; 
        span.textContent = `✅ GENUINE ${pct}%`;
    } else if (label === 'FAKE') {
        span.style.backgroundColor = isVerified ? '#ff9800' : '#d32f2f'; 
        span.textContent = `🚫 FAKE ${pct}%`;
    } else if (label === 'HUMAN') {
        span.style.backgroundColor = '#2e7d32'; 
        span.textContent = `✅ HUMAN ${pct}%`;
    } else if (label === 'AI' || label === 'BOT') {
        span.style.backgroundColor = '#d32f2f'; 
        span.textContent = `🤖 AI ${pct}%`;
    } else {
        span.style.backgroundColor = '#9e9e9e'; 
        span.textContent = `⚖️ UNCERTAIN ${pct}%`;
    }
    return span;
  }

  function attachBadge(node, label, confidence, isVerified, text) {
    if (node.getAttribute('data-rg-status') === 'done') return;
    if (node.querySelector('.rg-badge')) {
        node.setAttribute('data-rg-status', 'done');
        return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'rg-wrapper';
    wrapper.style.cssText = 'display:inline-flex; align-items:center; gap:5px; margin: 5px 0;';

    const badge = createBadge(label, confidence, isVerified);
    wrapper.appendChild(badge);

    // Only add "Why?" button for Amazon reviews for now (can expand later)
    if (SITE_TYPE === 'AMAZON' && label !== 'ERR') {
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
    } else {
        if (SITE_TYPE === 'YOUTUBE') node.parentElement.insertBefore(wrapper, node);
        else if (SITE_TYPE === 'TWITTER') node.parentElement.appendChild(wrapper);
    }
    node.setAttribute('data-rg-status', 'done');
  }

  // --- 4. OPTIMIZED BATCH PROCESSING LOOP ---
  let processing = false;
  
  async function runAnalysis() {
    if (processing) return;
    processing = true;
    updateButton("⏳ Scanning...");

    if (SITE_TYPE === 'AMAZON') {
        try { document.querySelectorAll('.a-expander-header a').forEach(btn => btn.click()); } catch(e) {}
    }

    const items = getItemsToAnalyze();
    const newItems = items.filter(i => !i.getAttribute('data-rg-status'));

    if (newItems.length === 0) {
        updateButton("✅ No New Items");
        setTimeout(() => updateButton("🔍 Scan Page"), 2000);
        processing = false;
        return;
    }

    // Prepare Batches
    const BATCH_SIZE = 5; 
    const textBuffer = [];
    const nodeBuffer = [];

    // 1. Gather Valid Text
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

    // 2. Process in Chunks
    for (let i = 0; i < textBuffer.length; i += BATCH_SIZE) {
        const texts = textBuffer.slice(i, i + BATCH_SIZE);
        const nodes = nodeBuffer.slice(i, i + BATCH_SIZE);
        
        const endpoint = SITE_TYPE === 'AMAZON' ? PREDICT_BATCH_URL : PREDICT_COMMENT_BATCH_URL;

        try {
            // SINGLE NETWORK CALL per batch
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ texts: texts })
            });
            const data = await res.json();

            // Map results back to nodes
            if (data.results) {
                data.results.forEach((result, idx) => {
                    const node = nodes[idx];
                    const text = texts[idx];
                    const isVerified = SITE_TYPE === 'AMAZON' && node.innerText.includes('Verified Purchase');
                    attachBadge(node, result.label, result.confidence, isVerified, text);
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
    if (SITE_TYPE !== 'AMAZON') { btn.style.background = '#1DA1F2'; btn.style.borderColor = 'transparent'; }
    btn.onmouseover = () => btn.style.transform = 'scale(1.05)';
    btn.onmouseout = () => btn.style.transform = 'scale(1)';
    btn.onclick = runAnalysis;
    document.body.appendChild(btn);
  }

  injectButton();
  setInterval(injectButton, 1000);

})();