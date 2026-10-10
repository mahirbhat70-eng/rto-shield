# FlipPrice AI — Merchant Bleed Audit Report
**Prepared for:** Aesthetic Threads D2C  
**Generated:** 2026-10-08  
**Decision Engine:** FlipPrice v2.0 (D3-calibrated Logistic Regression)  

---

## 1. Executive Summary

| Metric | Your Cohort Value | Note |
| :--- | :--- | :--- |
| **Total Analyzed Orders** | **300** | Extracted from raw merchant export |
| **COD Order Volume** | **300 (100.0%)** | Subject to return & fraud risk |
| **Status Quo RTO Bleed** | **₹12,300.00** | At industry standard ₹150 logistics loss / return |
| **FlipPrice Measured Savings** | **₹256.44** | **Net margin protected on your orders** |

> **Key Takeaway:** A static pincode blocklist would kill 0 orders in your cohort, forfeiting precious margins. FlipPrice safely keeps 0 of those orders alive as real revenue.

---

## 2. Recommended Action Distribution

Instead of blunt binary blocking, FlipPrice prices the exact friction of every action:

| Action | Orders Routed | Percentage | Operational Flow |
| :--- | :--- | :--- | :--- |
| **ALLOW_COD** | 238 | 79.3% | Frictionless dispatch (low risk) |
| **VERIFY_ADDRESS** | 62 | 20.7% | Automated ₹2 WhatsApp OTP address confirm |
| **REQUIRE_DEPOSIT** | 0 | 0.0% | ₹100 commitment advance before dispatch |
| **FORCE_PREPAID** | 0 | 0.0% | COD disabled (extreme risk) |

---

## 3. Static Pincode Blocklist vs. FlipPrice Decision Pricing

Comparison against a static blacklist app (e.g. KLIP / Shopify Blocklist) on your orders:

| Threshold (τ) | Pincodes Blocked | Good Buyers Killed | Bad Orders Blocked | Net ₹ Impact If Orders Lost | Required Conversion to Break-Even |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **τ = 0.2** | 0 | 0 | 0 | **−₹0.00** | **0%** |
| **τ = 0.25** | 0 | 0 | 0 | **−₹0.00** | **0%** |
| **τ = 0.3** | 0 | 0 | 0 | **−₹0.00** | **0%** |
| **τ = 0.35** | 0 | 0 | 0 | **−₹0.00** | **0%** |

### The Opportunity Cost Trap
- At τ = 0.25, a static blocklist eliminates **0** orders.
- But doing so kills **0 good customers**, costing you far more in gross margin than you save in return freight.
- **FlipPrice safely preserves 0 of those orders** using calibrated WhatsApp verifications and partial deposits.

---

## 4. Sample Flagged Orders & Prescribed Interventions

| Order ID | Pincode | Category | Value | Risk Score | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `#2026` | `700001` | Apparel | ₹454 | **0.295** | `VERIFY_ADDRESS` |
| `#2151` | `700001` | Apparel | ₹451 | **0.295** | `VERIFY_ADDRESS` |
| `#2277` | `560038` | Apparel | ₹589 | **0.295** | `VERIFY_ADDRESS` |
| `#2015` | `700001` | Apparel | ₹628 | **0.294** | `VERIFY_ADDRESS` |
| `#2201` | `400001` | Apparel | ₹664 | **0.294** | `VERIFY_ADDRESS` |
| `#2077` | `600001` | Apparel | ₹474 | **0.294** | `VERIFY_ADDRESS` |
| `#2290` | `560038` | Apparel | ₹743 | **0.294** | `VERIFY_ADDRESS` |
| `#2276` | `400001` | Apparel | ₹534 | **0.293** | `VERIFY_ADDRESS` |
| `#2257` | `600001` | Apparel | ₹789 | **0.293** | `VERIFY_ADDRESS` |
| `#2136` | `600001` | Apparel | ₹550 | **0.293** | `VERIFY_ADDRESS` |

---
*Calibration Status: No outcome labels in CSV; calibrated using pre-trained prior distribution.*  
*Report generated strictly from merchant-provided data. No fabricated case studies.*
