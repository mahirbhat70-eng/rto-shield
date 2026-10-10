# FlipPrice AI — Pilot-to-Proof Operations Playbook (Wave 5)

## Executive Summary
This playbook governs the commercial execution of **Phase 5: Pilot-to-Proof**. It defines the merchant qualification criteria, pre-registered measurement methodology, weekly value reporting cadence, churn mitigation, and review collection mechanics required to convert pilot brands into paying subscribers and case studies.

---

## 1. Ideal Customer Profile (ICP) & 200-Brand Segmentation (P-01)
### Target Qualifications:
* **Monthly GMV:** ₹10,00,000 to ₹1,00,00,000 (1,000 to 20,000 orders/month).
* **Payment Method Mix:** COD share $\ge 50\%$.
* **Current RTO Rate:** Baseline COD return rate $\ge 22\%$.
* **E-Commerce Stack:** Shopify or Shopify Plus with courier aggregation (Delhivery, Shiprocket, BlueDart).

### Priority Category Breakdown:
1. **Apparel & Ethnic Wear (40% of target list):** High COD share (65–75%), heavy sizing ambiguity, high gross margin (45–55%). *Prime candidate for VERIFY_ADDRESS.*
2. **Footwear & Sneakers (25% of target list):** Bulky forward/reverse shipping costs (₹180–₹240/order), high fit rejection rate.
3. **Beauty & Personal Care (20% of target list):** Fast replenishment, high repeat LTV. *Zero tolerance for customer churn from blunt blocklists.*
4. **Home & Decor (15% of target list):** High AOV (₹2,500–₹6,000), severe fragile return damage costs.

---

## 2. The 30-Day Zero-Risk Pilot Offer (P-02)
* **Contract Structure:** 30 days of free access on the Growth Tier (up to 10,000 orders).
* **The "Margin Guarantee":** If FlipPrice AI does not deliver at least ₹25,000 in net measured savings over the 30-day pilot, the merchant owes ₹0 and receives a complimentary catalog elasticity analysis.
* **Merchant Commitment:**
  - Share past 90-day order CSV for pre-period calibration.
  - Keep FlipPrice order tagging active for 30 consecutive days.
  - Participate in a 20-minute post-pilot review call and grant case-study rights upon positive ROI.

---

## 3. Pre-Registered Measurement Protocol (P-03)
To ensure that savings claims are undisputable, the measurement protocol is locked **before** day 1 of traffic:

### Method A: Pincode-Week Switchback Randomized Trial (Recommended for >5k orders/mo)
* Pincodes are partitioned into matched clusters based on historical delivery volume.
* Every 7 days, clusters rotate between:
  - **Control Arm:** Status quo (unconditional dispatch or legacy blocklist).
  - **Treatment Arm:** FlipPrice AI multi-action decision pricing.
* **Estimator:** Augmented Inverse Probability Weighting (AIPW) comparing delivery success and net margin per cluster-week.

### Method B: Synthetic Difference-in-Differences (SDID) (For <5k orders/mo)
* Baseline period: Preceding 60 days of merchant order history.
* Test period: 30 days live on FlipPrice.
* Compares the treatment cohort against an external market trend benchmark to control for macro festive or logistics drift.

---

## 4. 7-Day Pilot Onboarding Runbook (P-05)

| Day | Milestone | Objective | Owner |
| :---: | :--- | :--- | :--- |
| **Day 0** | **App Install & OAuth** | Merchant installs app; tenant generated in SQLite; webhook registered. | Merchant / Ops Lead |
| **Day 1** | **Historical Bleed Audit** | Ingest 90-day CSV export; deliver executive Bleed Report showing exact ₹ baseline. | FlipPrice Lead |
| **Day 2** | **Shadow Mode Scoring** | Score 100% of incoming orders silently without mutating tags; verify latency & webhooks. | Tech Lead |
| **Day 3** | **Staged Traffic: 20% Live** | Activate tags on 20% of orders; review warehouse pick-pack team comfort with `fp_action`. | Merchant Ops |
| **Day 7** | **100% Full Cutover** | Full decision pricing active; first automated Weekly Value Digest dispatched. | Automated Cron |

---

## 5. Churn Interview & Objection Defusal Script (P-06)
If a pilot merchant pauses tagging or indicates churn risk:
1. **Root Cause Diagnosis:**
   - *"Did the warehouse team face confusion with order tags?"* $\rightarrow$ Connect with courier mapping adapter.
   - *"Did customers question the ₹100 deposit?"* $\rightarrow$ Review threshold tuning; adjust deposit floor to top 5% extreme risk.
   - *"Was WhatsApp template delivery delayed?"* $\rightarrow$ Switch provider routing between Gupshup and Wati.
2. **The Counter-Offer:** Offer an additional 14 days with customized merchant margin settings keyed to their specific product categories.

---

## 6. Expansion Playbook: Quarterly Policy Re-Optimization Upsell (P-07)
* After 90 days of live serving, merchant catalogs and customer profiles experience distribution drift.
* FlipPrice runs [`scripts/reoptimize_policy.py`](file:///c:/Users/vansh/OneDrive/Desktop/rto-shield-prep/rto-shield/scripts/reoptimize_policy.py) on quarterly order logs.
* **Upsell Value:** Re-optimizing intervention parameters ($rto\_red$, $succ\_drop$, category margins) captures an additional **+12% to +18% uplift**, justifying expansion from Starter to Growth plans.

---

## 7. Shopify App Store Review Collector Workflow (P-08)
* **Trigger Milestone:** When a merchant's cumulative savings pass **₹25,000** in the SQLite audit ledger.
* **Automated Notification:** In-app admin banner:
  > *"🎉 FlipPrice has protected ₹25,400 in net margin for {{merchant_name}} this month! Love the results? Share your experience with a 30-second review on the Shopify App Store."*
* **Direct Deep-Link:** Deep-links directly to the review submission modal on the Shopify App Store listing.
