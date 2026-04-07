"""
xai_service.py — v7.1 (Matrix Edition with Pragmatic Nuance)

XAI pipeline:
  1. Receives raw text, DeBERTa label (Style + Authorship), and confidence.
  2. Feeds context to Groq (LLaMA 3) with a strict 2x2 matrix system prompt.
  3. Groq natively cross-references the Style axis with the Authorship axis.
  4. Streams a single, coherent sentence back to the frontend.
  5. Pure-Python template fallback if the API fails.
"""

import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Module-level state ──────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
_WORD_DELAY  = 0.04

def load_explainer(app_model, app_tokenizer):
    """
    Kept for compatibility with app.py's boot sequence.
    """
    print("[XAI] Explainer loaded (Pure LLM Matrix Mode).")

# ──────────────────────────────────────────────────────────────────
# GROQ NARRATION
# ──────────────────────────────────────────────────────────────────

def _groq_narrate(text: str, label: str, confidence: float) -> str:
    """Ask Groq to reverse-engineer the primary model's decision across all 4 quadrants."""
    if not GROQ_API_KEY:
        raise ValueError("No GROQ_API_KEY set.")

    system = (
        "You are the Explainable AI (XAI) module for an e-commerce review fraud detector. "
        "Your job is to explain WHY the primary AI model assigned a specific label to a review. "
        "The model classifies reviews across a 2x2 matrix: Style (Genuine vs. Promotional) and Authorship (Human vs. AI).\n\n"
        "Use this strict framework to reverse-engineer the decision based on the text:\n"
        "1. Promotional-style: Look for rigid formatting, sterile/generic template praise lacking personal context, marketing buzzwords, or 'seller voice'.\n"
        "2. Genuine-style: Look for nuanced trade-offs, specific personal context, OR pragmatic/terse buyer observations (e.g., blunt, everyday language typical of genuine regional shoppers).\n"
        "3. AI-assisted: Look for overly polished syntax, sterile/perfect grammar, robotic transitions ('Furthermore', 'Overall'), and cliché LLM phrases ('game changer').\n"
        "4. Human-written: Look for natural conversational pacing, slang, emotional nuance, minor typos, or highly specific niche use-cases.\n\n"
        "Keep your explanation under 3 sentences. Be direct and analytical. Do not use generic filler.\n"
        "Format exactly as: 'AI Logic: [Your explanation]'"
    )

    user_prompt = f"Review Text: '{text}'\nModel Label: {label} ({confidence*100:.0f}% confidence)\nExplain why the model made this decision."

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json={
            "model":      GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens":  85,
        },
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ──────────────────────────────────────────────────────────────────
# TEMPLATE FALLBACK
# ──────────────────────────────────────────────────────────────────

def _template_fallback(label: str, confidence: float) -> str:
    """Pure-Python fallback when Groq is unavailable. Respects the full 2-part label."""
    if confidence >= 0.88:
        return f"AI Logic: Flagged as {label} ({confidence:.0%}) due to strong stylistic and syntactic markers typical of this category."
    elif confidence >= 0.70:
        return f"AI Logic: Classified as {label} ({confidence:.0%}) based on the overall formatting, tone, and pacing of the writing."
    else:
        return f"AI Logic: Classified as {label} with moderate confidence ({confidence:.0%}). Writing patterns lean this way, but signals are mixed."

# ──────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────

def stream_explanation(text: str, label: str, confidence: float):
    """
    Main entry point called from app.py /explain_stream.
    Streams the explanation word-by-word.
    """
    print(f"[XAI] Explaining: {label} @ {confidence:.2%}")

    explanation = ""
    try:
        explanation = _groq_narrate(text, label, confidence)
    except Exception as e:
        print(f"[XAI] Groq failed, using template. Error: {e}")
        explanation = _template_fallback(label, confidence)

    # Ensure the frontend gets the expected prefix formatting
    if not explanation.startswith("AI Logic:"):
        explanation = f"AI Logic: {explanation}"

    # Stream word-by-word
    words = explanation.split()
    for i, word in enumerate(words):
        yield ("" if i == 0 else " ") + word
        time.sleep(_WORD_DELAY)