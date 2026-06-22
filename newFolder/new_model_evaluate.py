"""
threshold_analysis.py — ReviewGuard Live Threshold Determination
================================================================
Drop this file alongside live_shadow_run.csv in your isolated folder.
Run:  python threshold_analysis.py

What this does:
  1. Runs fresh inference with temperature calibration applied correctly
  2. Analyses the score distribution to find natural decision boundaries
  3. Computes optimal thresholds using multiple methods (Youden's J,
     cost-sensitive, percentile-based)
  4. Simulates every candidate threshold so you can see the exact
     Promotional / Uncertain / Genuine split before committing
  5. Detects domain shift signals (multilingual, short reviews, etc.)
  6. Saves a full report CSV and a distribution plot
  7. Prints the exact lines to paste into app.py

NOTE: The script does NOT require ground truth labels.
It uses unsupervised distribution analysis + principled cost assumptions.
"""

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
from tqdm import tqdm

# ── PATHS — edit only these two lines ────────────────────────────────────────
CSV_PATH   = "live_shadow_run.csv"
MODEL_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "model"))

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
MAX_LEN        = 128
BATCH_SIZE     = 32
CALIBRATION_T  = 1.2375   # Must match app.py exactly

# ── COST ASSUMPTION (asymmetric error costs) ──────────────────────────────────
# Cost of a false positive (calling genuine review "Promotional")
# relative to cost of a false negative (missing a fake review).
# 3.0 = false positives are 3x more damaging than false negatives.
# Rationale: wrongly accusing a genuine reviewer destroys user trust;
#            missing a fake review leaves the user no worse than before.
FP_COST        = 3.0

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Mirror of app.py clean_live_inference_text — must stay in sync."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\d+\.\d+ out of 5 stars', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reviewed in [a-zA-Z\s]+ on \d{1,2} [a-zA-Z]+ \d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Verified Purchase', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Colour:\s*[a-zA-Z0-9\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Size:\s*[a-zA-Z0-9\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+ people found this helpful', '', text, flags=re.IGNORECASE)
    text = re.sub(r'One person found this helpful', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bHelpful\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bReport\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def apply_temperature_calibration(raw_fake: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling in log-odds space.
    This is the EXACT same operation as _calibrated_probs() in app.py.
    Always apply this ONCE to raw softmax outputs.
    """
    if abs(T - 1.0) < 1e-4:
        return raw_fake
    raw_real    = 1.0 - raw_fake
    log_odds    = np.log(raw_fake / (raw_real + 1e-9))
    scaled      = log_odds / T
    return 1.0 / (1.0 + np.exp(-scaled))


def is_non_english(text: str) -> bool:
    """Rough proxy: contains non-ASCII characters (Hindi/Tamil/etc.)."""
    return any(ord(c) > 127 for c in text)


def classify_at_threshold(fake_probs: np.ndarray,
                           fake_thresh: float,
                           genuine_thresh: float) -> np.ndarray:
    """
    Returns array of labels: 2=Promotional, 1=Uncertain, 0=Genuine.
    genuine_thresh is the FAKE probability below which we call it Genuine,
    i.e. real_prob = 1 - fake_prob >= (1 - genuine_thresh).
    """
    labels = np.ones(len(fake_probs), dtype=int)          # default: Uncertain
    labels[fake_probs >= fake_thresh]     = 2              # Promotional
    labels[fake_probs <= genuine_thresh]  = 0              # Genuine
    return labels


def simulate_threshold(fake_probs, fake_t, genuine_t, label=""):
    labels   = classify_at_threshold(fake_probs, fake_t, genuine_t)
    n        = len(fake_probs)
    promo    = (labels == 2).sum()
    unc      = (labels == 1).sum()
    gen      = (labels == 0).sum()
    print(f"  {label:<35} Promo={promo:4d} ({100*promo/n:.1f}%)  "
          f"Uncertain={unc:4d} ({100*unc/n:.1f}%)  "
          f"Genuine={gen:4d} ({100*gen/n:.1f}%)")
    return promo, unc, gen


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  ReviewGuard — Live Threshold Analysis")
print("="*70)

if not os.path.exists(CSV_PATH):
    sys.exit(f"ERROR: {CSV_PATH} not found. Place it in the same folder.")

df = pd.read_csv(CSV_PATH).dropna(subset=['review_text'])
df['review_text'] = df['review_text'].astype(str)
df['cleaned']     = df['review_text'].apply(clean_text)

# Flag short and non-English reviews for domain shift reporting
df['word_count']      = df['cleaned'].str.split().str.len()
df['is_short']        = df['word_count'] < 5        # below inference threshold
df['is_non_english']  = df['review_text'].apply(is_non_english)

print(f"\nLoaded  : {len(df)} reviews")
print(f"Too short to infer (<5 words)  : {df['is_short'].sum()}")
print(f"Non-English (non-ASCII)        : {df['is_non_english'].sum()}")

valid_df   = df[~df['is_short']].copy()
short_df   = df[df['is_short']].copy()
print(f"Valid for inference            : {len(valid_df)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL INFERENCE (raw softmax, no calibration yet)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading model from: {MODEL_DIR}")
if not os.path.exists(MODEL_DIR):
    sys.exit(f"ERROR: Model directory not found at {MODEL_DIR}")

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
model     = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR)
print("Model loaded.\n")

raw_fake_probs = []
raw_real_probs = []

texts = valid_df['cleaned'].tolist()

print("Running inference (raw softmax — calibration applied separately)...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    inputs = tokenizer(
        batch,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    logits = model(inputs).logits
    probs  = tf.nn.softmax(logits, axis=1).numpy()

    # IMPORTANT: Confirm your index mapping.
    # From training: label_binary=0 → Fake(CG), label_binary=1 → Genuine(OR)
    # So model output index 0 = Fake probability, index 1 = Genuine probability.
    for p in probs:
        raw_fake_probs.append(float(p[0]))   # index 0 = Fake
        raw_real_probs.append(float(p[1]))   # index 1 = Genuine

raw_fake_arr = np.array(raw_fake_probs)
raw_real_arr = np.array(raw_real_probs)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CALIBRATION (applied ONCE, cleanly)
# ═══════════════════════════════════════════════════════════════════════════════

cal_fake_arr = apply_temperature_calibration(raw_fake_arr, CALIBRATION_T)
cal_real_arr = 1.0 - cal_fake_arr

print("\n--- Calibration Effect ---")
print(f"  Raw    | mean={raw_fake_arr.mean():.4f}  median={np.median(raw_fake_arr):.4f}  "
      f"std={raw_fake_arr.std():.4f}")
print(f"  Cal T={CALIBRATION_T} | mean={cal_fake_arr.mean():.4f}  "
      f"median={np.median(cal_fake_arr):.4f}  std={cal_fake_arr.std():.4f}")

# Check if the uploaded CSV already has calibrated scores
if 'fake_prob' in df.columns:
    existing = df.loc[~df['is_short'], 'fake_prob'].values
    raw_diff = np.abs(existing - raw_fake_arr).mean()
    cal_diff = np.abs(existing - cal_fake_arr).mean()
    print(f"\n  Comparing CSV fake_prob to this run:")
    print(f"    Mean absolute diff vs raw    : {raw_diff:.4f}")
    print(f"    Mean absolute diff vs cal    : {cal_diff:.4f}")
    if cal_diff < raw_diff:
        print("  ✓ CSV scores appear to be POST-calibration (cal_diff smaller)")
        print("    → app.py CALIBRATION_T should stay at 1.2375 and NOT re-apply to these.")
    else:
        print("  ✓ CSV scores appear to be PRE-calibration (raw_diff smaller)")
        print("    → Calibration is being applied fresh. This is correct.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Score Distribution (calibrated fake probability) ---")
percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]
for p in percentiles:
    v = np.percentile(cal_fake_arr, p)
    print(f"  {p:3d}th percentile : {v:.4f}")

print("\n--- Bin Counts ---")
bins   = [0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.001]
labels = ['<0.40','0.40-0.50','0.50-0.60','0.60-0.70',
          '0.70-0.80','0.80-0.85','0.85-0.90','0.90-0.95','≥0.95']
for label, low, high in zip(labels, bins[:-1], bins[1:]):
    count = ((cal_fake_arr >= low) & (cal_fake_arr < high)).sum()
    print(f"  {label:12s} : {count:4d}  ({100*count/len(cal_fake_arr):.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OPTIMAL THRESHOLD METHODS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  THRESHOLD METHODS")
print("="*70)

# ── Method A: Youden's J (unsupervised proxy) ─────────────────────────────────
# Without ground truth labels, we approximate using the score distribution.
# We treat the lower 25th percentile as a "likely genuine" proxy population
# and the upper 75th percentile as a "likely fake" proxy population,
# then find the threshold that maximises separation between them.
proxy_genuine = cal_fake_arr[cal_fake_arr <= np.percentile(cal_fake_arr, 25)]
proxy_fake    = cal_fake_arr[cal_fake_arr >= np.percentile(cal_fake_arr, 75)]

candidate_thresholds = np.linspace(0.40, 0.99, 500)
best_j, best_t_j = -1, 0.5

for t in candidate_thresholds:
    tpr = (proxy_fake    >= t).mean()   # true positive rate on proxy fakes
    tnr = (proxy_genuine <  t).mean()   # true negative rate on proxy genuines
    j   = tpr + tnr - 1
    if j > best_j:
        best_j, best_t_j = j, t

print(f"\nMethod A — Youden's J (distribution proxy)")
print(f"  Optimal fake threshold : {best_t_j:.4f}  (J={best_j:.4f})")


# ── Method B: Cost-sensitive threshold ───────────────────────────────────────
# Minimise expected cost = FP_COST * FP_rate + 1.0 * FN_rate
best_cost, best_t_cost = float('inf'), 0.5

for t in candidate_thresholds:
    fp_rate = (proxy_genuine >= t).mean()
    fn_rate = (proxy_fake    <  t).mean()
    cost    = FP_COST * fp_rate + 1.0 * fn_rate
    if cost < best_cost:
        best_cost, best_t_cost = cost, t

print(f"\nMethod B — Cost-sensitive (FP_COST={FP_COST}x)")
print(f"  Optimal fake threshold : {best_t_cost:.4f}  (cost={best_cost:.4f})")
print(f"  Rationale: false positives penalised {FP_COST}x more than false negatives")
print(f"  (Adjust FP_COST at top of script to explore other cost assumptions)")


# ── Method C: Natural valley (GMM-inspired) ──────────────────────────────────
# Find the lowest-density point in the score distribution between 0.40 and 0.98.
# This is where a threshold causes the fewest classification reversals.
hist_vals, hist_edges = np.histogram(cal_fake_arr, bins=100, density=True)
# Look for minimum density in the middle region (0.30 to 0.95)
mid_mask = (hist_edges[:-1] >= 0.30) & (hist_edges[:-1] <= 0.95)
mid_vals  = hist_vals[mid_mask]
mid_edges = hist_edges[:-1][mid_mask]

if len(mid_vals) > 0:
    valley_idx = np.argmin(mid_vals)
    best_t_valley = float(mid_edges[valley_idx])
else:
    best_t_valley = best_t_cost

print(f"\nMethod C — Natural valley (lowest density point)")
print(f"  Valley at fake threshold : {best_t_valley:.4f}")
print(f"  (This is where fewest reviews sit — safest cut point)")


# ── Method D: Percentile anchoring ───────────────────────────────────────────
# Flag only the top N% as promotional. Choose N based on prior belief
# about what fraction of Amazon reviews are genuinely fake.
# Research estimates: 10-30% of Amazon reviews are inauthentic.
# We use 20% as a conservative upper bound.
TARGET_FAKE_RATE = 0.20
t_percentile = float(np.percentile(cal_fake_arr, 100 * (1 - TARGET_FAKE_RATE)))

print(f"\nMethod D — Percentile anchoring (target fake rate={TARGET_FAKE_RATE*100:.0f}%)")
print(f"  Fake threshold to flag top {TARGET_FAKE_RATE*100:.0f}% : {t_percentile:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THRESHOLD SIMULATION TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  THRESHOLD SIMULATION (calibrated scores, genuine_thresh = 1 - fake_thresh)")
print("="*70)
print(f"  {'Threshold config':<35} {'Promotional':>12}  {'Uncertain':>12}  {'Genuine':>10}")
print("-"*70)

# Genuine threshold = symmetric by default (1 - fake_thresh)
# This means the uncertain band widens as fake_thresh rises.
candidates = [
    ("Current app.py (T=0.60)",    0.60, 0.40),
    ("Youden's J",                  round(best_t_j,    2), round(1-best_t_j,    2)),
    ("Cost-sensitive (FP=3x)",      round(best_t_cost, 2), round(1-best_t_cost, 2)),
    ("Natural valley",              round(best_t_valley,2),round(1-best_t_valley,2)),
    (f"Percentile (top {TARGET_FAKE_RATE*100:.0f}%)", round(t_percentile,2), round(1-t_percentile,2)),
    ("Conservative (T=0.90)",       0.90, 0.40),
    ("Recommended (T=0.92)",        0.92, 0.40),
    ("Strict (T=0.95)",             0.95, 0.40),
]

results = {}
for name, ft, gt in candidates:
    p, u, g = simulate_threshold(cal_fake_arr, ft, gt, label=name)
    results[name] = (ft, gt, p, u, g)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DOMAIN SHIFT REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  DOMAIN SHIFT ANALYSIS")
print("="*70)

# Assign calibrated scores back to valid_df
valid_df = valid_df.copy()
valid_df['raw_fake_prob'] = raw_fake_arr
valid_df['cal_fake_prob'] = cal_fake_arr

# Non-English reviews — what scores do they get?
non_eng = valid_df[valid_df['is_non_english']]
eng     = valid_df[~valid_df['is_non_english']]

print(f"\nNon-English reviews  : {len(non_eng)}")
if len(non_eng) > 0:
    print(f"  Mean fake prob (cal): {non_eng['cal_fake_prob'].mean():.4f}")
    print(f"  % scored ≥ 0.60     : {(non_eng['cal_fake_prob'] >= 0.60).mean()*100:.1f}%")
    print(f"  % scored ≥ 0.92     : {(non_eng['cal_fake_prob'] >= 0.92).mean()*100:.1f}%")

print(f"\nEnglish reviews      : {len(eng)}")
if len(eng) > 0:
    print(f"  Mean fake prob (cal): {eng['cal_fake_prob'].mean():.4f}")
    print(f"  % scored ≥ 0.60     : {(eng['cal_fake_prob'] >= 0.60).mean()*100:.1f}%")
    print(f"  % scored ≥ 0.92     : {(eng['cal_fake_prob'] >= 0.92).mean()*100:.1f}%")

# Short reviews (5–15 words) — common in Indian e-commerce
short_valid = valid_df[valid_df['word_count'] <= 15]
long_valid  = valid_df[valid_df['word_count'] > 15]
print(f"\nShort reviews (5-15 words) : {len(short_valid)}")
if len(short_valid) > 0:
    print(f"  Mean fake prob (cal)     : {short_valid['cal_fake_prob'].mean():.4f}")
    print(f"  % scored ≥ 0.92          : {(short_valid['cal_fake_prob'] >= 0.92).mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE RESULTS CSV
# ═══════════════════════════════════════════════════════════════════════════════

# Reconstruct full df with scores (short reviews get NaN)
output_df = df.copy()
output_df['raw_fake_prob'] = np.nan
output_df['cal_fake_prob'] = np.nan
output_df.loc[~output_df['is_short'], 'raw_fake_prob'] = raw_fake_arr
output_df.loc[~output_df['is_short'], 'cal_fake_prob'] = cal_fake_arr

# Apply recommended threshold labels
for name, (ft, gt, *_) in results.items():
    col = f"label_{name.split('(')[0].strip().lower().replace(' ','_')[:20]}"
    output_df[col] = "SKIPPED"
    mask = ~output_df['is_short']
    preds = classify_at_threshold(
        output_df.loc[mask, 'cal_fake_prob'].values, ft, gt
    )
    label_map = {2: "Promotional", 1: "Uncertain", 0: "Genuine"}
    output_df.loc[mask, col] = [label_map[p] for p in preds]

output_df.to_csv("threshold_analysis_results.csv", index=False)
print(f"\n✅ Full results saved → threshold_analysis_results.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PLOT
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ReviewGuard — Live Threshold Analysis", fontweight="bold", fontsize=13)

# Left: distribution with threshold lines
ax = axes[0]
ax.hist(cal_fake_arr, bins=80, color="#c0392b", alpha=0.7, label="All reviews")
ax.hist(cal_fake_arr[valid_df['is_non_english'].values], bins=40,
        color="#2980b9", alpha=0.6, label="Non-English reviews")
ax.axvline(0.60,         color="black",   linestyle="--",  lw=1.5, label="Current (0.60)")
ax.axvline(best_t_cost,  color="#e67e22", linestyle="-.",  lw=1.5, label=f"Cost-sensitive ({best_t_cost:.2f})")
ax.axvline(t_percentile, color="#27ae60", linestyle=":",   lw=1.5, label=f"Percentile top 20% ({t_percentile:.2f})")
ax.axvline(0.92,         color="#8e44ad", linestyle="-",   lw=2.0, label="Recommended (0.92)")
ax.set_xlabel("Calibrated Fake Probability")
ax.set_ylabel("Number of Reviews")
ax.set_title("Score Distribution with Candidate Thresholds")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Right: promotional% at each threshold
ax2    = axes[1]
thresh_range = np.linspace(0.50, 0.99, 100)
promo_pcts   = [(cal_fake_arr >= t).mean() * 100 for t in thresh_range]
ax2.plot(thresh_range, promo_pcts, color="#c0392b", lw=2)
ax2.axvline(0.60,  color="black",   linestyle="--", lw=1.5, label="Current (0.60)")
ax2.axvline(0.92,  color="#8e44ad", linestyle="-",  lw=2.0, label="Recommended (0.92)")
ax2.axhline(20,    color="#27ae60", linestyle=":",  lw=1.2, label="20% fake prior")
ax2.axhline(30,    color="#e67e22", linestyle=":",  lw=1.2, label="30% fake prior")
ax2.set_xlabel("Fake Threshold")
ax2.set_ylabel("% Reviews Labelled Promotional")
ax2.set_title("Promotional Rate vs Threshold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("threshold_analysis_plot.png", dpi=150)
print("📊 Plot saved → threshold_analysis_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FINAL RECOMMENDATION + APP.PY PATCH
# ═══════════════════════════════════════════════════════════════════════════════

# Pick the most conservative of the data-driven methods
recommended_fake    = max(best_t_cost, t_percentile, 0.90)
recommended_genuine = 0.40   # asymmetric: wider uncertain band on the genuine side

promo_pct = (cal_fake_arr >= recommended_fake).mean() * 100

print("\n" + "="*70)
print("  FINAL RECOMMENDATION")
print("="*70)
print(f"\n  Recommended fake threshold    : {recommended_fake:.4f}")
print(f"  Recommended genuine threshold : ≤{recommended_genuine:.4f} (fake_prob)")
print(f"  Promotional rate at this T    : {promo_pct:.1f}% of live reviews")
print(f"  Uncertain band                : {recommended_genuine:.2f} – {recommended_fake:.2f}")

print(f"""
  ┌─ Paste this into app.py get_combined_assessment() ─────────────────┐
  │                                                                      │
  │   FAKE_THRESHOLD    = {recommended_fake:.4f}   # data-driven, shadow run      │
  │   GENUINE_THRESHOLD = {recommended_genuine:.4f}   # asymmetric (FP cost > FN)   │
  │                                                                      │
  │   if fake_prob >= FAKE_THRESHOLD:                                    │
  │       style      = "Promotional-style"                               │
  │       style_conf = min(fake_prob, 0.98)                              │
  │   elif fake_prob <= GENUINE_THRESHOLD:                               │
  │       style      = "Genuine-style"                                   │
  │       style_conf = min(1.0 - fake_prob, 0.98)                        │
  │   else:                                                              │
  │       style      = "Uncertain-style"                                 │
  │       style_conf = max(fake_prob, 1.0 - fake_prob)                   │
  └──────────────────────────────────────────────────────────────────────┘
""")

print("  CALIBRATION NOTE:")
print(f"  The scores above are computed with CALIBRATION_T={CALIBRATION_T}.")
print("  Ensure app.py applies calibration ONCE on raw softmax output.")
print("  If your CSV fake_prob values are already calibrated, do not")
print("  re-calibrate them — set CALIBRATION_T=1.0 for CSV comparison only.\n")