# Case Study: How Aesthetic Threads D2C Salvaged ₹84,200 in Net Margin and Cut COD Return Rates by 26.5%

*Pilot Partner: Aesthetic Threads (Contemporary Ethnic Wear, Jaipur/Bengaluru)*  
*Monthly Volume: ~6,200 Orders | COD Share: 68% | Average Order Value: ₹1,650*  
*Study Window: 30-Day Staged Pilot Evaluation*

---

## Executive Summary
Prior to deploying FlipPrice AI, Aesthetic Threads suffered from an unsustainable 29.4% Cash on Delivery return rate, incurring over ₹1.85 lakh per month in unrecoverable reverse logistics fees. In an attempt to stem losses, the brand implemented a static pincode blacklist across 450+ Tier-3 pin codes. 

While reverse logistics costs declined, **monthly revenue collapsed by ₹4.2 lakh** because genuine customers in developing corridors were blocked from ordering.

By deploying FlipPrice AI's multi-action decision pricing via native Shopify webhooks, Aesthetic Threads replaced the static blacklist with dynamic order-level intervention routing:
* **COD Return Rate:** Dropped from **29.4% to 21.6%** (−26.5% relative decrease).
* **Genuine Customer Sales Salvaged:** **1,620 orders** previously rejected by the blacklist were kept alive as completed deliveries.
* **Net Monthly Margin Recovered:** **₹84,200** after all operational WhatsApp and gateway verification friction.

---

## The Challenge: The "Pincode Blacklist Trap"
Like many growing Indian D2C apparel brands, Aesthetic Threads operates on healthy gross merchandise margins (~48%), but fragile net margins after marketing and logistics. 

> *"We were bleeding on reverse shipping from eastern and northern corridors,"* explains the Head of E-Commerce. *"Our courier aggregation tool advised us to turn off COD for 450 pincodes. What they didn't tell us was that 70% of shoppers in those pincodes were completely legitimate buyers with zero history of returns. We were setting fire to profitable customer relationships."*

---

## The FlipPrice AI Solution
Aesthetic Threads installed the FlipPrice AI Shopify App with zero theme code modifications. The 30-day pilot was structured in three phases:

1. **Week 1 (Shadow Mode):** FlipPrice scored 100% of incoming orders silently. The engine demonstrated that 82% of orders flagged as "high risk" by the legacy blacklist were in fact profitable to ship under simple WhatsApp address verification.
2. **Weeks 2–3 (Staged Rollout):** 
   - **Low-Risk Orders (21%):** `ALLOW_COD` $\rightarrow$ Immediate frictionless dispatch.
   - **Borderline Orders (56%):** `VERIFY_ADDRESS` $\rightarrow$ Dispatched automated WhatsApp confirmation with 1-click confirm/cancel buttons (₹2 cost).
   - **Severe-Risk Orders (23%):** `REQUIRE_DEPOSIT` $\rightarrow$ Fulfillment auto-held pending a ₹100 commitment deposit link via UPI.
3. **Week 4 (Full Optimization):** Catalog margins were customized: higher margin kurtas (52%) were routed more permissively than low-margin accessories (25%).

---

## Quantitative Results (30-Day Comparison)

| Metric | Pre-Pilot Baseline (Static Blocklist) | Post-Pilot (FlipPrice AI) | Business Impact |
| :--- | :---: | :---: | :---: |
| **Total COD Orders Received** | 4,216 | 4,280 | +1.5% checkout confidence |
| **Orders Blocked at Checkout** | 895 (21.2%) | **0 (0.0%)** | **Zero customer alienation** |
| **COD Return Rate (RTO)** | 29.4% | **21.6%** | **−7.8 percentage points** |
| **Total Returned Shipments** | 976 | 694 | **282 fewer returns** |
| **Forfeited Customer Margin** | ₹1,12,000 | **₹0** | **₹1.12L gross margin preserved** |
| **Friction Spend (WhatsApp + PG)** | ₹0 | ₹9,820 | Low operational cost |
| **Net Financial Value Unlocked** | — | **+₹84,200** | **8.4× ROI on software fee** |

---

## Operational Takeaways
1. **The ₹100 Deposit Sifted Fraud from Intent:** 44% of shoppers prompted for a ₹100 deposit paid immediately via UPI. Of those who paid the deposit, **98.2% accepted delivery**. Fraudulent bulk buyers and casual orderers abandoned without incurring forward logistics fees.
2. **Warehouse Operations Simplified:** Pick-and-pack fulfillment teams simply monitored the `fp_action` order tag inside Shopify Admin. Orders with `fp_hold:DEPOSIT_REQUIRED` remained paused automatically until UPI settlement.
3. **Zero Developer Overhead:** The entire integration was live within 5 minutes of Shopify App Store installation.

---

## Executive Testimonial
> *"FlipPrice AI fundamentally changed how we think about risk. We used to treat RTO as a binary crime that required banning pincodes. FlipPrice showed us it’s an optimization problem. Our returns are down, our top-line revenue is up, and we didn't have to rewrite a single line of checkout code."*  
> — **Rohan V., Co-Founder & COO, Aesthetic Threads**
