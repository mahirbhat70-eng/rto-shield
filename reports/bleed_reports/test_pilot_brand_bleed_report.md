# FlipPrice AI — Merchant Bleed Audit Report
**Prepared for:** Test Pilot Brand  
**Generated:** 2026-10-08  
**Decision Engine:** FlipPrice v2.0 (D3-calibrated Logistic Regression)  

---

## 1. Executive Summary

| Metric | Your Cohort Value | Note |
| :--- | :--- | :--- |
| **Total Analyzed Orders** | **250** | Extracted from raw merchant export |
| **COD Order Volume** | **250 (100.0%)** | Subject to return & fraud risk |
| **Status Quo RTO Bleed** | **₹10,200.00** | At industry standard ₹150 logistics loss / return |
| **FlipPrice Measured Savings** | **₹342.29** | **Net margin protected on your orders** |

> **Key Takeaway:** A static pincode blocklist would kill 0 orders in your cohort, forfeiting precious margins. FlipPrice safely keeps 0 of those orders alive as real revenue.

---

## 2. Recommended Action Distribution

Instead of blunt binary blocking, FlipPrice prices the exact friction of every action:

| Action | Orders Routed | Percentage | Operational Flow |
| :--- | :--- | :--- | :--- |
| **ALLOW_COD** | 165 | 66.0% | Frictionless dispatch (low risk) |
| **VERIFY_ADDRESS** | 85 | 34.0% | Automated ₹2 WhatsApp OTP address confirm |
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
| `#1036` | `110001` | Apparel | ₹553 | **0.296** | `VERIFY_ADDRESS` |
| `#1041` | `560038` | Apparel | ₹561 | **0.296** | `VERIFY_ADDRESS` |
| `#1022` | `110001` | Apparel | ₹565 | **0.296** | `VERIFY_ADDRESS` |
| `#1171` | `400001` | Apparel | ₹512 | **0.295** | `VERIFY_ADDRESS` |
| `#1138` | `110001` | Apparel | ₹767 | **0.295** | `VERIFY_ADDRESS` |
| `#1125` | `500001` | Apparel | ₹858 | **0.294** | `VERIFY_ADDRESS` |
| `#1218` | `400001` | Apparel | ₹860 | **0.294** | `VERIFY_ADDRESS` |
| `#1124` | `400001` | Apparel | ₹889 | **0.294** | `VERIFY_ADDRESS` |
| `#1088` | `500001` | Apparel | ₹662 | **0.294** | `VERIFY_ADDRESS` |
| `#1015` | `500001` | Apparel | ₹913 | **0.294** | `VERIFY_ADDRESS` |

---
*Calibration Status: No outcome labels in CSV; calibrated using pre-trained prior distribution.*  
*Report generated strictly from merchant-provided data. No fabricated case studies.*
