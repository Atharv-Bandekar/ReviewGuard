// content.js - Optimized Batch Processing + UX Enhancements

(() => {
    const API_BASE = "http://127.0.0.1:8000";
    const PREDICT_BATCH_URL = `${API_BASE}/predict_batch`; 
    const EXPLAIN_STREAM_URL = `${API_BASE}/explain_stream`;

    if (!window.location.hostname.includes('amazon')) return;

    // --- ENHANCEMENT 2: Analytics State ---
    let pageStats = { 
        genuineHuman: 0, genuineAI: 0, promoHuman: 0, promoAI: 0, uncertain: 0, total: 0 
    };

    // Listen for Popup requesting stats
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "GET_STATS") sendResponse(pageStats);
    });

    // --- ENHANCEMENT 3: Local State Caching ---
    function getCacheKey(text) {
        // Creates a unique, safe key based on the text content and length
        const clean = text.replace(/[^a-zA-Z0-9]/g, '').substring(0, 40);
        return `rg_${clean}_${text.length}`;
    }

    // --- 1. SELECTORS & EXTRACTION ---
    function getItemsToAnalyze() {
        const nodes = Array.from(document.querySelectorAll('div[id^="customer_review-"]'));
        return nodes.filter(n => n.offsetParent !== null);
    }

    function extractText(node) {
        let rawText = "";

        // 1. Try the direct path first
        const standardBox = node.querySelector('[data-hook="review-body"] span') || 
                            node.querySelector('.review-text-content span');
        
        if (standardBox) {
            rawText = standardBox.innerText;
        } else {
            // 2. RUTHLESS FALLBACK: Clone and destroy noise
            const clone = node.cloneNode(true);
            
            const noiseSelectors = [
                '.a-profile', '.review-date', '.video-block', '.review-title', 
                '.review-comments', '.review-format-strip', '.cr-helpful-text', 
                '.cr-helpful-button'
            ];
            
            noiseSelectors.forEach(s => 
                clone.querySelectorAll(s).forEach(n => n.remove())
            );
            
            rawText = clone.innerText;
        }
        
        // 3. UNIVERSAL CLEANUP (Applied to everything)
        if (!rawText) return "";
        
        // Added '\.?' to catch the periods that often follow these sentences
        let text = rawText.replace(/\d+\s+people\s+found\s+this\s+helpful\.?/gi, '');
        text = text.replace(/One\s+person\s+found\s+this\s+helpful\.?/gi, '');
        text = text.replace(/Read more|Helpful|Report/gi, '');
        
        // Collapse multiple spaces/newlines into a single space
        return text.replace(/\s{2,}/g, ' ').trim();
    }

    // --- 2. UI HELPERS (Badges & Explanations) ---
    async function fetchExplanation(text, label, confidence, container) {
        try {
            container.innerHTML = '<b>AI Logic:</b> <span class="ai-stream-text"></span>';
            const streamTarget = container.querySelector('.ai-stream-text');
            container.style.display = 'block';

            let borderColor = '#43a047'; 
            if (label.includes('Promotional-style, AI-assisted')) borderColor = '#b71c1c'; 
            else if (label.includes('Promotional-style')) borderColor = '#e53935'; 
            else if (label.includes('AI-assisted')) borderColor = '#fbc02d'; 
            container.style.borderLeft = `3px solid ${borderColor}`;

            // FIX: Safely pass the full review text in the body via POST
            const payload = {
                text: text, 
                label: label,
                confidence: confidence
            };

            const response = await fetch(EXPLAIN_STREAM_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                cache: 'no-store'
            });

            if (!response.body) throw new Error("Stream not supported");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                streamTarget.textContent += decoder.decode(value, { stream: true });
                container.scrollTop = container.scrollHeight;
            }
        } catch (e) {
            container.innerHTML = 'Error connecting to XAI engine.';
        }
    }

   function createBadge(label, confidence) {
        const span = document.createElement('span');
        span.className = 'rg-badge';
        span.style.cssText = 'display:inline-block; padding:2px 6px; margin:0 5px; border-radius:4px; font-weight:700; font-size:11px; color:white;';
        const pct = (confidence * 100).toFixed(0);

        // FIX: Decoupled string matching to catch "Possibly AI-assisted"
        if (label.includes('Genuine-style') && label.includes('Human-written')) {
            span.style.backgroundColor = '#2e7d32'; span.textContent = `✅ ${label} ${pct}%`;
        } else if (label.includes('Genuine-style') && label.includes('AI-assisted')) {
            span.style.backgroundColor = '#fbc02d'; span.style.color = '#000'; span.textContent = `⚠️ ${label} ${pct}%`;
        } else if (label.includes('Promotional-style') && label.includes('Human-written')) {
            span.style.backgroundColor = '#d32f2f'; span.textContent = `🚫 ${label} ${pct}%`;
        } else if (label.includes('Promotional-style') && label.includes('AI-assisted')) {
            span.style.backgroundColor = '#b71c1c'; span.textContent = `🚫 ${label} ${pct}%`;
        } else {
            span.style.backgroundColor = '#9e9e9e'; span.textContent = `⚖️ ${label} ${pct}%`;
        }
        
        return span;
    }

    function attachBadge(node, label, confidence, text) {
        if (node.getAttribute('data-rg-status') === 'done') return;

       // Update Stats
        pageStats.total++;
        if (label.includes('Genuine-style') && label.includes('Human-written')) pageStats.genuineHuman++;
        else if (label.includes('Genuine-style') && label.includes('AI-assisted')) pageStats.genuineAI++;
        else if (label.includes('Promotional-style') && label.includes('Human-written')) pageStats.promoHuman++;
        else if (label.includes('Promotional-style') && label.includes('AI-assisted')) pageStats.promoAI++;
        else pageStats.uncertain++;

        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display:inline-flex; align-items:center; gap:5px; margin: 5px 0;';
        wrapper.appendChild(createBadge(label, confidence));

        const btn = document.createElement('button');
        btn.textContent = '💡 Why?';
        btn.style.cssText = 'border:1px solid #ccc; background:#fff; cursor:pointer; font-size:11px; padding:2px 8px; border-radius:10px; color:#333;';
        
        const explainBox = document.createElement('div');
        explainBox.style.cssText = 'display:none; margin-top:5px; font-size:13px; color:#333; background:#f0f2f5; padding:8px; border-radius:4px; line-height: 1.4;';

        btn.onclick = (e) => { e.preventDefault(); btn.style.display = 'none'; fetchExplanation(text, label, confidence, explainBox); };
        wrapper.appendChild(btn);
        
        const header = node.querySelector('.a-profile') || node.querySelector('.review-header');
        if (header) { header.parentElement.insertBefore(wrapper, header.nextSibling); wrapper.insertAdjacentElement('afterend', explainBox); } 
        else { node.prepend(wrapper); wrapper.insertAdjacentElement('afterend', explainBox); }
        
        node.setAttribute('data-rg-status', 'done');
        updateTrustWidget(); // Refresh widget
    }

    // --- ENHANCEMENT 1: Product Trust Score Widget ---
    function updateTrustWidget() {
        if (pageStats.total === 0) return;
        
        let widget = document.getElementById('rg-trust-widget');
        const titleTarget = document.getElementById('titleSection') || document.getElementById('title');
        
        if (!widget && titleTarget) {
            widget = document.createElement('div');
            widget.id = 'rg-trust-widget';
            widget.style.cssText = 'margin-top: 10px; padding: 12px; background: #f8f9fa; border-left: 4px solid #232f3e; border-radius: 4px; font-family: sans-serif; display: flex; align-items: center; gap: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);';
            titleTarget.parentElement.insertBefore(widget, titleTarget.nextSibling);
        }
        
        if (widget) {
            const genuinePct = Math.round(((pageStats.genuineHuman + pageStats.genuineAI) / pageStats.total) * 100);
            const isGood = genuinePct >= 70;
            
            widget.innerHTML = `
                <div style="font-size: 24px;">🛡️</div>
                <div>
                    <div style="font-weight: bold; font-size: 14px; color: #0f1111;">ReviewGuard Trust Score: <span style="color: ${isGood ? '#2e7d32' : '#d32f2f'}; font-size: 16px;">${genuinePct}% Genuine</span></div>
                    <div style="font-size: 12px; color: #565959;">Based on ${pageStats.total} scanned reviews (${pageStats.promoHuman + pageStats.promoAI} flagged as promotional/spam).</div>
                </div>
            `;
        }
    }

    // --- 3. BATCH PROCESSING ---
    let processing = false;
    async function runAnalysis() {
        if (processing) return;
        processing = true;
        updateButton("⏳ Scanning...");

        const items = getItemsToAnalyze().filter(i => !i.getAttribute('data-rg-status'));
        if (items.length === 0) { updateButton("✅ Done"); setTimeout(() => updateButton("🔍 Scan Page"), 2000); processing = false; return; }

        const BATCH_SIZE = 5; 
        const textBuffer = [], nodeBuffer = [];

        items.forEach(node => {
            const text = extractText(node);
            if (text && text.length > 5) {
                const cacheKey = getCacheKey(text);
                const cached = sessionStorage.getItem(cacheKey);
                
                if (cached) {
                    // CACHE HIT: Instant render!
                    const data = JSON.parse(cached);
                    attachBadge(node, data.label, data.confidence, text);
                } else {
                    // CACHE MISS: Send to backend
                    textBuffer.push(text);
                    nodeBuffer.push({ node, text, cacheKey });
                    node.setAttribute('data-rg-status', 'pending');
                }
            } else {
                node.setAttribute('data-rg-status', 'done'); 
            }
        });

        for (let i = 0; i < textBuffer.length; i += BATCH_SIZE) {
            const texts = textBuffer.slice(i, i + BATCH_SIZE);
            const nodesChunk = nodeBuffer.slice(i, i + BATCH_SIZE);
            
            try {
                const res = await fetch(PREDICT_BATCH_URL, {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ texts: texts })
                });
                const data = await res.json();

                if (data.results) {
                    data.results.forEach((result, idx) => {
                        const target = nodesChunk[idx];
                        // Save to cache
                        sessionStorage.setItem(target.cacheKey, JSON.stringify(result));
                        attachBadge(target.node, result.label, result.confidence, target.text);
                    });
                }
            } catch (e) {
                console.error("Batch Error:", e);
                nodesChunk.forEach(t => t.node.removeAttribute('data-rg-status'));
            }
        }

        updateButton("🔍 Scan More");
        processing = false;
    }

    // --- 4. FLOATING BUTTON ---
    function updateButton(text) { const btn = document.getElementById('rg-float-btn'); if (btn) btn.textContent = text; }
    function injectButton() {
        if (document.getElementById('rg-float-btn')) return;
        const btn = document.createElement('button');
        btn.id = 'rg-float-btn'; btn.textContent = '🔍 Scan Page';
        btn.style.cssText = `position: fixed; bottom: 20px; right: 20px; z-index: 2147483647; padding: 12px 20px; background: #232f3e; color: #fff; border: 2px solid #fff; border-radius: 30px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.3);`;
        btn.onclick = runAnalysis;
        document.body.appendChild(btn);
    }
    injectButton(); setInterval(injectButton, 1000);
})();