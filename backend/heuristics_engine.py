"""
heuristics_engine.py — v2

Changes from v1:
  - Burstiness check now uses Coefficient of Variation (CV) instead of
    raw variance. CV is length-normalised, so it doesn't fire on long
    well-written reviews that happen to have uniform sentence lengths.
    Guard tightened: requires num_sentences >= 5 (was 4) and CV < 0.30.
  - All other logic (pronoun deficit, lexical complexity, TTR, XAI-only
    signals) is unchanged — they were well-designed in v1.
  - _AI_SCORE_MAX constant unchanged (1.05); normalisation preserved.

What this file does NOT do (unchanged from v1):
  - Does not adjust DeBERTa's raw_fake_prob / raw_real_prob.
  - Style verdict is owned entirely by DeBERTa.
  - Heuristics are evidence for the XAI layer, not judges.
"""

import re
from textblob import TextBlob

_AI_SCORE_MAX = 1.05


def analyze_text_heuristics(text, raw_fake_prob, raw_real_prob):
    """
    Structural analysis layer. Computes AI-likelihood and XAI triggers.

    Parameters
    ----------
    text          : str   -- the raw review text
    raw_fake_prob : float -- DeBERTa P(promotional)  [not modified]
    raw_real_prob : float -- DeBERTa P(genuine)       [not modified]

    Returns
    -------
    raw_fake_prob : float -- returned unchanged
    raw_real_prob : float -- returned unchanged
    ai_score      : float -- Axis-2 likelihood (0–0.95)
    triggers      : list[dict]
    """
    if not text or len(text.strip()) < 5:
        return raw_fake_prob, raw_real_prob, 0.0, [
            {"category": "Human", "text": "Text too short to analyze"}
        ]

    words         = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    num_words     = max(len(words), 1)
    sentences     = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = max(len(sentences), 1)

    raw_ai_score = 0.0
    triggers     = []

    # ── 1. Pronoun deficit ──────────────────────────────────────────
    first_person = {'i', 'me', 'my', 'mine', 'we', 'our', 'us'}
    pronoun_count = sum(1 for w in words if w in first_person)

    if pronoun_count == 0 and num_words > 30:
        raw_ai_score += 0.35
        triggers.append({"category": "AI", "text": "Zero personal pronouns"})
    elif pronoun_count < 2 and num_words > 50:
        raw_ai_score += 0.15
        triggers.append({"category": "AI", "text": "Very few personal pronouns"})

    # ── 2. Lexical complexity ───────────────────────────────────────
    complex_words      = sum(1 for w in words if len(w) >= 7)
    complex_word_ratio = complex_words / num_words
    avg_word_length    = sum(len(w) for w in words) / num_words

    if avg_word_length > 5.2 and complex_word_ratio > 0.20:
        raw_ai_score += 0.30
        triggers.append({
            "category": "AI",
            "text": (
                f"Overly complex vocabulary "
                f"({complex_word_ratio:.0%} of words ≥7 chars, "
                f"avg word length {avg_word_length:.1f})"
            ),
        })

    # ── 3. Burstiness (CV-based) ────────────────────────────────────
    # FIX from v1: raw variance was not length-normalised. A 200-word
    # genuine review with consistent 12-word sentences fired the trigger
    # even though variance=18 is reasonable at that mean length.
    # Coefficient of Variation (CV = std/mean) is scale-independent.
    # CV < 0.30 means sentences are unnaturally uniform regardless of length.
    if num_sentences >= 5:
        lengths   = [len(s.split()) for s in sentences]
        mean_len  = sum(lengths) / num_sentences
        variance  = sum((l - mean_len) ** 2 for l in lengths) / num_sentences
        std_len   = variance ** 0.5
        cv        = std_len / (mean_len + 1e-9)

        if cv < 0.30 and mean_len > 8:
            raw_ai_score += 0.25
            triggers.append({
                "category": "AI",
                "text": (
                    f"Unnaturally uniform sentence pacing "
                    f"(CV={cv:.2f}, mean {mean_len:.1f} words/sentence)"
                ),
            })

    # ── 4. Type-Token Ratio ─────────────────────────────────────────
    ttr = len(set(words)) / num_words
    if num_words > 80 and ttr < 0.55:
        raw_ai_score += 0.15
        triggers.append({"category": "AI", "text": "Repetitive word usage"})

    # Normalise before capping
    ai_score = min(raw_ai_score / _AI_SCORE_MAX, 0.95)

    # ── 5. XAI-only signals (do not modify probabilities) ───────────

    # 5a. Negative sentiment
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.10:
        triggers.append({
            "category": "Human",
            "text": "Genuine negative sentiment detected",
        })

    # 5b. Hedging language
    hedging_words = {
        'but', 'however', 'although', 'though', 'except',
        'unfortunately', 'despite', 'yet', 'still', 'otherwise',
    }
    hedge_count = sum(1 for w in words if w in hedging_words)
    if hedge_count >= 2:
        triggers.append({
            "category": "Human",
            "text": f"Hedging/qualifying language detected ({hedge_count} instances)",
        })

    # 5c. Hype formatting
    original_words      = text.split()
    caps_ratio          = (
        sum(1 for w in original_words if w.isupper() and len(w) > 2)
        / max(len(original_words), 1)
    )
    exclamation_density = text.count('!') / num_sentences

    if num_words > 10 and (exclamation_density > 1.0 or caps_ratio > 0.10):
        triggers.append({
            "category": "Promo",
            "text": "Aggressive hype formatting (all-caps/exclamations)",
        })

    # ── 6. Fallback ─────────────────────────────────────────────────
    if not triggers:
        if raw_fake_prob > 0.50:
            triggers.append({
                "category": "Promo",
                "text": "Model-identified promotional patterns",
            })
        else:
            triggers.append({
                "category": "Human",
                "text": "Standard pacing and vocabulary",
            })

    return raw_fake_prob, raw_real_prob, ai_score, triggers
