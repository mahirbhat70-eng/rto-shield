# RTO Shield — Decision Console (Live Dashboard)

Deploy guide + verification record for `dashboard.py`, the judge-facing interactive
console built for your video recording and submission link.

---

## 1. What you got

| File | Purpose |
|---|---|
| `dashboard.py` | The whole console. Goes in your repo **root**, next to `app.py` |
| `config.toml` | Dark theme. Goes to `.streamlit/config.toml` in your repo |
| `screenshots/` | Verified renders of all 4 views |

Four views, mapped to your pitch script acts:

| View | Pitch act | What it shows live |
|---|---|---|
| 01 · Command Center | Act 1 + Act 3 | Hero "We don't predict the coin flip. We price it." + ₹71,741 / ₹69,786 / 2.0× / 94.7% cards + how-pricing-works strip |
| 02 · Live Decision Engine | Act 4 | Presets → SCORE → P(RTO) dial, **expected-loss price menu with ROUTED·ARGMIN highlight**, TreeSHAP bars, latency chip, audit trail with tamper-evident fingerprint + JSONL download |
| 03 · Policy Frontier | Act 2 | The threshold-vs-loss sweep **recomputed live** (bitwise-identical to frozen stage-4): binary VERIFY/PREPAID curves, argmin line, tuned thresholds 0.20/0.48 marked |
| 04 · Portfolio Evidence | Act 3 | Expected vs realized P&L, action donut, Monte Carlo band, P/R, per-action calibration, pytest provenance |

---

## 2. Safety of the frozen repo (read this before committing)

- **Purely additive**: two new files. `app.py`, `src/`, `tests/`, `reports/`, `requirements.txt` — zero bytes changed.
- **Tests re-run on the exact frozen checkout (93f1157 + these two files): 60/60 passed (31s).**
- No test imports `dashboard.py` → CI outcome unchanged (green stays green).
- No new dependency: Altair ships with Streamlit 1.40.2; `requirements.txt` untouched.
- Dashboard reads frozen artifacts through the same `src/serve/scorer.py` your verified demo uses — it cannot drift from the repo numbers.
- Rollback anytime: switch the Cloud app's main file back to `app.py` (both apps live in the repo; the old scorer demo is untouched and still deployable).

---

## 3. Deploy (5 minutes, same URL as your existing demo)

1. Copy files into your local clone:
   - `dashboard.py` → repo root
   - `config.toml` → `.streamlit/config.toml`
2. Commit + push:
   ```bash
   git add dashboard.py .streamlit/config.toml
   git commit -m "feat: live Decision Console dashboard (additive serving-layer UI)"
   git push origin main
   ```
3. Go to share.streamlit.io → your `rto-shield` app → **Settings → Main file path**:
   change `app.py` → `dashboard.py` → **Save** (app reboots, URL unchanged:
   the one already in your README badge).
4. First boot takes ~2–4 min on Cloud (models + SHAP). Every visit after that is fast.

Local test before pushing (optional): `streamlit run dashboard.py` from repo root.

---

## 4. Verified numbers (what you will see on screen)

All re-verified live in this build, on the frozen artifacts:

- VERIFY preset (₹852): **P(RTO) 39.4%**, decision **VERIFY_ADDRESS**, price menu
  ALLOW −₹44.00 · **VERIFY −₹54.60 (ROUTED·ARGMIN)** · DEPOSIT −₹50.07 · PREPAID −₹4.32,
  routing saves ₹10.59/order vs always-allow. (Identical to your logged-out demo test.)
- ALLOW preset → ALLOW_COD @ 21.3%. DEPOSIT preset → REQUIRE_DEPOSIT. PREPAID method → passthrough card.
- Policy Frontier (val_cal COD, 3,673 orders): binary VERIFY best −₹3,08,740 @ t=0.20,
  binary PREPAID best −₹2,90,924 @ t=0.48, argmin routing −₹3,27,071, edge +₹18,331 (5.9%).
  Vectorized sweep verified **bitwise-identical** (`maxdiff 0.00e+00`) to the frozen loop in
  `src/eval/stage4_evaluate.py`; thresholds 0.20/0.48 reproduced.
- Latency chip measures the real full path incl. TreeSHAP (observed 11–68 ms locally;
  Cloud typically ~40–60 ms — comfortably inside the "<100 ms" narration; do NOT say
  "sub-15ms", say "under 100 milliseconds").

---

## 5. Recording cheat-sheet (120s script mapping)

- **Act 1 (0:00–0:20)**: Command Center hero. Say the tagline, point at 28.27% bleed card.
- **Act 2 (0:20–0:42)**: Policy Frontier. "I swept every threshold from 0.01 to 0.99 — both
  binary curves stay above the red line. That's why there's no threshold to tune."
  The white dots = the tuned thresholds 0.20/0.48.
- **Act 3 (0:42–1:00)**: Portfolio Evidence (or Command Center cards): ₹71,741 expected →
  ₹69,786 realized (−2.7%), 2.0×, 13.1% uplift, P(savings>0)=100%.
- **Act 4 (1:00–1:48)**: Live Decision Engine → click **VERIFY** (auto-scores!) → walk
  P(RTO) → price menu (−₹54.60 highlighted, say "negative fifty-four rupees sixty") →
  SHAP ("COD charge and pincode history are pushing the price up") → audit receipt
  ("sha256 fingerprint, tamper-evident, downloadable") → latency chip ("under 100 ms,
  full path including explanations").
- **Act 5 (1:48–2:00)**: back to Command Center hero, close on the tagline.

Tip: on first load of each view Streamlit may show a short spinner — open all 4 views
once before recording so everything is warm, then reload and record.
