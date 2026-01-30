// content.js - Final Hybrid Version (Deep Amazon Analysis + Fast Social Scanning)

(() => {
  const API_BASE = "http://127.0.0.1:8000";
  const PREDICT_REVIEW_URL = `${API_BASE}/predict`;         
  const PREDICT_COMMENT_URL = `${API_BASE}/predict_comment`; 
  const EXPLAIN_URL = `${API_BASE}/explain`;

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
  async function fetchExplanation(text, label, confidence, container) {
    try {
        container.innerHTML = '<span class="loading-spinner">↻</span> <i>Asking AI...</i>';
        const res = await fetch(EXPLAIN_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, label, confidence })
        });
        const data = await res.json();
        
        if (data.explanation) {
            container.innerHTML = `<b>AI Logic:</b> ${data.explanation}`;
            container.style.borderLeft = `3px solid ${label === 'FAKE' ? '#e53935' : '#43a047'}`;
        } else {
            container.innerHTML = 'Could not generate explanation.';
        }
    } catch (e) {
        container.innerHTML = 'Error connecting to AI.';
    }
  }

  function createBadge(label, confidence, isVerified) {
    const span = document.createElement('span');
    span.className = 'rg-badge';
    span.style.cssText = 'display:inline-block; padding:2px 6px; margin:0 5px; border-radius:4px; font-weight:700; font-size:11px; color:white; vertical-align:middle; font-family:sans-serif; z-index:9999;';
    
    const pct = (confidence * 100).toFixed(0);

    // --- 1. AMAZON LABELS (Genuine / Fake) ---
    if (label === 'GENUINE') {
        span.style.backgroundColor = '#2e7d32'; // Green
        span.textContent = `✅ GENUINE ${pct}%`;
    } 
    else if (label === 'FAKE') {
        // If it's a Verified Purchase but detected as Fake, use Orange (Warning)
        // Otherwise use Red (Danger)
        span.style.backgroundColor = isVerified ? '#ff9800' : '#d32f2f'; 
        span.textContent = `🚫 FAKE ${pct}%`;
    }

    // --- 2. SOCIAL LABELS (Human / AI) ---
    else if (label === 'HUMAN') {
        span.style.backgroundColor = '#2e7d32'; // Green
        span.textContent = `✅ HUMAN ${pct}%`;
    } 
    else if (label === 'BOT') {
        span.style.backgroundColor = '#d32f2f'; // Red
        span.textContent = `🤖 AI ${pct}%`; // Display as "AI" per your request
    }

    // --- 3. FALLBACK / UNCERTAIN ---
    else {
        span.style.backgroundColor = '#9e9e9e'; // Grey
        span.textContent = `⚖️ UNCERTAIN ${pct}%`;
    }

    return span;
}

  // --- 4. INJECTION LOGIC ---
  function attachBadge(node, label, confidence, isVerified, text) {
    if (node.getAttribute('data-rg-status') === 'done') return;
    
    // Safety check for existing badges
    if (node.querySelector('.rg-badge') || (node.parentElement && node.parentElement.querySelector('.rg-badge'))) {
        node.setAttribute('data-rg-status', 'done');
        return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'rg-wrapper';
    wrapper.style.cssText = 'display:inline-flex; align-items:center; gap:5px; margin: 5px 0;';

    const badge = createBadge(label, confidence, isVerified);
    wrapper.appendChild(badge);

    // 💡 AMAZON EXCLUSIVE: Add "Why?" Button
    if (SITE_TYPE === 'AMAZON' && label !== 'ERR') {
        const btn = document.createElement('button');
        btn.textContent = '💡 Why?';
        btn.style.cssText = 'border:1px solid #ccc; background:#fff; cursor:pointer; font-size:11px; padding:2px 8px; border-radius:10px; color:#333;';
        
        const explainBox = document.createElement('div');
        explainBox.style.cssText = 'display:none; margin-top:5px; font-size:13px; color:#333; background:#f0f2f5; padding:8px; border-radius:4px; width:100%;';

        btn.onclick = (e) => {
            e.preventDefault();
            btn.style.display = 'none'; 
            explainBox.style.display = 'block'; 
            fetchExplanation(text, label, confidence, explainBox);
        };
        wrapper.appendChild(btn);
        
        // Insert Wrapper + ExplainBox for Amazon
        const header = node.querySelector('.a-profile') || node.querySelector('.review-header');
        if (header) {
             header.parentElement.insertBefore(wrapper, header.nextSibling);
             wrapper.insertAdjacentElement('afterend', explainBox);
        } else {
             node.prepend(wrapper);
             wrapper.insertAdjacentElement('afterend', explainBox);
        }
    } 
    // 🚀 SOCIAL MEDIA: Badge Only (Fast)
    else {
        if (SITE_TYPE === 'YOUTUBE') {
            node.parentElement.insertBefore(wrapper, node);
        } else if (SITE_TYPE === 'TWITTER') {
            node.parentElement.appendChild(wrapper);
        }
    }

    node.setAttribute('data-rg-status', 'done');
  }

  // --- 5. PROCESSING LOOP ---
  let processing = false;
  
  async function runAnalysis() {
    if (processing) return;
    processing = true;
    updateButton("⏳ Scanning...");

    if (SITE_TYPE === 'AMAZON') {
        try { document.querySelectorAll('.a-expander-header a').forEach(btn => btn.click()); } catch(e) {}
    }

    const items = getItemsToAnalyze();
    console.log(`[ReviewGuard] Found ${items.length} items on ${SITE_TYPE}`);

    const newItems = items.filter(i => !i.getAttribute('data-rg-status'));

    if (newItems.length === 0) {
        updateButton("✅ No New Items");
        setTimeout(() => updateButton("🔍 Scan Page"), 2000);
        processing = false;
        return;
    }

    const BATCH_SIZE = 5; 
    for (let i = 0; i < newItems.length; i += BATCH_SIZE) {
        const chunk = newItems.slice(i, i + BATCH_SIZE);
        
        await Promise.all(chunk.map(async (node) => {
            node.setAttribute('data-rg-status', 'pending');
            const text = extractText(node);
            
            if (!text || text.length < 5) {
                node.setAttribute('data-rg-status', 'done');
                return; 
            }

            const url = SITE_TYPE === 'AMAZON' ? PREDICT_REVIEW_URL : PREDICT_COMMENT_URL;
            
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text: text })
                });
                const data = await res.json();
                const isVerified = SITE_TYPE === 'AMAZON' && node.innerText.includes('Verified Purchase');
                
                // Pass text so we can use it for explanation if needed
                attachBadge(node, data.label, data.confidence, isVerified, text);
            } catch (e) {
                console.error(e);
                node.removeAttribute('data-rg-status');
            }
        }));
    }

    updateButton("🔍 Scan More");
    processing = false;
  }

  // --- 6. FLOATING BUTTON ---
  function updateButton(text) {
    const btn = document.getElementById('rg-float-btn');
    if (btn) btn.textContent = text;
  }

  function injectButton() {
    if (document.getElementById('rg-float-btn')) return;

    const btn = document.createElement('button');
    btn.id = 'rg-float-btn';
    btn.textContent = '🔍 Scan Page';
    btn.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; z-index: 2147483647;
        padding: 12px 20px; background: #232f3e; color: #fff;
        border: 2px solid #fff; border-radius: 30px;
        font-family: sans-serif; font-weight: bold; cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    `;
    
    if (SITE_TYPE !== 'AMAZON') {
        btn.style.background = '#1DA1F2'; 
        btn.style.borderColor = 'transparent';
    }

    btn.onmouseover = () => btn.style.transform = 'scale(1.05)';
    btn.onmouseout = () => btn.style.transform = 'scale(1)';
    btn.onclick = runAnalysis;
    document.body.appendChild(btn);
  }

  injectButton();
  setInterval(injectButton, 1000);

})();