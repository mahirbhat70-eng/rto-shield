# RTO Shield — Pitch & Promo Video Script

> **Target Duration:** 2 minutes (120 seconds)  
> **Audience:** Razorpay Hackathon Judges / Senior FinTech & Risk Leadership  
> **Key Thesis:** Classification thresholds throw away merchant profit. RTO Shield is an expected-loss decision engine that delivers **2.0× higher savings** and **13.1% portfolio profit uplift** at **94.7% of the Bayes ceiling**.

---

## Video Storyboard & Timing Overview

| Time | Section | On-Screen Visual | Audio Goal |
|---|---|---|---|
| **0:00 – 0:20** | **The Bleed** | COD checkout animation; ₹120 RTO logistics loss vs ₹49 shipping margin | Hook with the real merchant pain: COD cancellation economics. |
| **0:20 – 0:45** | **The Flaw in Single Thresholds** | Binary threshold graph vs Multi-action cost matrix | Expose why 90% of solutions fail in production (false drop cost). |
| **0:45 – 1:15** | **The Mathematics & Results** | PR-AUC curve (94.7% Bayes) + 5,000-draw Monte Carlo distribution | Concrete numbers: ₹71,741 savings, 2.0× vs single threshold, 100% P(savings > 0). |
| **1:15 – 1:45** | **Live System Demo** | Live Streamlit UI + Real-time SHAP factors + JSONL Audit Export | Proof of execution: sub-15ms scoring, deterministic replay fingerprint. |
| **1:45 – 2:00** | **The Razorpay Impact** | Razorpay Magic Checkout integration architecture | Close on instant merchant ROI with zero customer checkout friction. |

---

## Word-for-Word Voiceover Script

### 0:00 – 0:20 | Act 1: The Indian E-Commerce Paradox
**Visual:** *Show title card: "RTO Shield — AI Risk Engine for Indian E-Commerce". Cut to an abandoned delivery package illustration or logistics tracking graphic.*

> *"In Indian e-commerce, Cash on Delivery is unavoidable — over 60% of consumers demand it. But for merchants, every Return-to-Origin is a direct financial loss: double shipping costs, reverse logistics, and dead inventory — averaging ₹120 to ₹180 per failed order.*  
> 
> *Most fraud detection systems treat this as a standard classification problem. If probability is above 0.5, they block the order. But doing that burns your highest-value customers."*

---

### 0:20 – 0:45 | Act 2: Why Binary Thresholds Fail
**Visual:** *Switch to the Decision Architecture diagram (`docs/decision_architecture.png`). Highlight the 4 distinct intervention tiers: ALLOW_COD, REQUIRE_DEPOSIT, VERIFY_ADDRESS, PREPAID_ONLY.*

> *"A ₹2,000 apparel order with a 30% risk of RTO shouldn't be blocked — because the 70% chance of a completed sale is worth far more than the delivery risk.  
> 
> That’s why we built **RTO Shield**. Instead of a blunt binary cutoff, RTO Shield calculates the **mathematical expected financial loss** across four granular interventions: Allow, Require a partial deposit, Verify the address via automated IVR/WhatsApp, or Convert to prepaid.*  
> 
> *By choosing the intervention that strictly minimizes expected loss, we protect the merchant's margin without alienating good buyers."*

---

### 0:45 – 1:15 | Act 3: Proven Numbers, Zero Leaks
**Visual:** *Display the Stage 5 benchmark charts and the Claim Matrix (`claim-matrix.md`). Pan across PR-AUC = 0.3313 (94.7% of Bayes Ceiling 0.3497) and Monte Carlo savings histogram.*

> *"We validated RTO Shield on a frozen, leak-free temporal split using calibrated LightGBM.  
> 
> The results speak for themselves:*
> - *We achieved **94.7% of the theoretical Bayes ceiling** on PR-AUC.*
> - *Our multi-action policy delivers **₹71,741 net savings** on our test cohort — **exactly 2.0 times higher** than the best tuned single-threshold baseline.*
> - *Across a 5,000-draw Monte Carlo simulation with noisy parameters, the probability of positive savings was **100%**.*
> - *All 60 automated tests in our suite pass with zero regressions.*"

---

### 1:15 – 1:45 | Act 4: Live Demonstration & Auditability
**Visual:** *Screen recording of the live Streamlit demo (`rto-shield-3qus23ktkhkzgqmxoytde5.streamlit.app`). Click the preset "VERIFY", show the green highlighted argmin in the Expected Loss table, show real-time SHAP waterfall drivers, then click 'Download Audit Trail (.jsonl)'.*

> *"Here it is running live on Streamlit Cloud.  
> 
> Notice how when an order comes in — say a ₹852 Home order with a 39% RTO probability — the engine doesn't blindly block it. The Expected Loss table reveals that automated address verification yields the lowest expected loss of -₹54.60.  
> 
> Each decision is scored in under 15 milliseconds, with local SHAP feature attributions explaining the exact drivers.  
> 
> And for compliance and dispute resolution, every single decision generates an immutable, tamper-evident SHA-256 replay fingerprint exported to a streaming JSONL audit trail."*

---

### 1:45 – 2:00 | Act 5: The Closing Proposition
**Visual:** *Return to presentation summary slide with GitHub link, live demo badge, and Razorpay logo overlay.*

> *"RTO Shield turns RTO management from an anxious guessing game into an exact financial optimization. It's ready to plug directly into Razorpay Magic Checkout or custom merchant gateways today.*  
> 
> *Explore our live app, inspect every number in our claim matrix, and run our test suite yourself. Thank you."*

---

## Technical Highlights for Judge Q&A / Accompanying Notes

- **Live Demo Link:** [https://rto-shield-3qus23ktkhkzgqmxoytde5.streamlit.app/](https://rto-shield-3qus23ktkhkzgqmxoytde5.streamlit.app/)
- **Full Q&A Repository:** [`docs/JUDGE_QA.md`](./JUDGE_QA.md) (covers all 10 toughest questions including leakage prevention, calibration error, and cost sensitivity).
- **Claim Matrix:** [`claim-matrix.md`](../claim-matrix.md) (maps every single metric to its exact pytest command).
