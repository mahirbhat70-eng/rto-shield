# FlipPrice AI — Shopify App Architecture Specification (v2.0)

## Executive Summary
The FlipPrice AI Shopify App serves as the frictionless merchant-facing layer over the FlipPrice Decision Pricing API. It delivers the **anti-blocklist advantage**: instead of rejecting customers outright on static pincode/phone blacklists, FlipPrice dynamically evaluates the expected monetary loss of each order and routes it to the cheapest safe intervention (`ALLOW_COD`, `VERIFY_ADDRESS`, `REQUIRE_DEPOSIT`, or `FORCE_PREPAID`).

---

## 1. Zero-Friction Integration Pattern (A-02)
Unlike legacy RTO blocklist tools that force custom Liquid/Theme edits or break standard Shopify checkout, FlipPrice operates entirely via native webhooks, order tags, and metafields:

```
[Shopify Checkout] 
       │
       ▼ (orders/create Webhook)
[FlipPrice App Gateway]
       │
       ├── 1. Cryptographic HMAC-SHA256 Verification (X-Shopify-Hmac-Sha256)
       ├── 2. Idempotency Cache Check (X-Shopify-Webhook-Id)
       ├── 3. Feature Mapping (DPDP Act 2023 salted customer hashing)
       │
       ▼
[FlipPrice Decision Engine API (/score)]
       │
       ├── Probability of Return p_rto (Logistic Regression D3 1-SE)
       ├── Expected Loss Argmin (Cost Engine with Rs 7 deposit, Rs 2 verify friction)
       │
       ▼
[Order Mutations Returned to Shopify]
       ├── Tags Applied: fp_risk:0.39, fp_action:VERIFY_ADDRESS, fp_el:48.20, fp_decision:PRICED
       ├── Metafields Written: namespace 'flipprice'
       │
       ▼
[Secondary Action Orchestration]
       ├── If VERIFY_ADDRESS  ──> Automated WhatsApp OTP / Address Check
       └── If REQUIRE_DEPOSIT ──> Auto-Hold Fulfillment Order + Send UPI Deposit Link
```

---

## 2. Dynamic Pincode COD Control (A-03, A-04)
Traditional tools offer a binary toggle: *"Turn off COD for pincode 845401"*. This forfeits genuine customer orders and brand goodwill.

FlipPrice provides an **EL-Break-Even Pincode Policy**:
- **Formula:** An order is structurally loss-making under unconditional dispatch only if $p_{\text{rto}} \times C_{\text{rto}} > (1 - p_{\text{rto}}) \times \text{margin}$.
- **Merchant Tooltip:** Explains in plain rupees why an intervention is required and how much margin is saved compared to blocking COD.
- **Example:** For an ₹1,800 apparel cart in a Tier-3 corridor with 28% historical RTO:
  - Static Blocklist: ₹0 profit (Order cancelled, customer lost).
  - Unconditional COD: −₹144 expected loss.
  - FlipPrice (`REQUIRE_DEPOSIT` ₹100): +₹216 net expected profit.

---

## 3. WhatsApp Interactive Confirmation (A-05)
For orders routed to `VERIFY_ADDRESS` (moderate risk):
- Dispatches a registered WhatsApp Business template (via Gupshup, Wati, or Interakt).
- Contains interactive buttons: `[Confirm Order (Free Shipping)]` and `[Cancel Order]`.
- Customer responses write back to order tags:
  - `CONFIRM` $\rightarrow$ `fp_wa_confirmed:true` (Release for packing).
  - `CANCEL` $\rightarrow$ `fp_wa_cancelled:true` (Cancel before shipping fee incurred).

---

## 4. Fulfillment Auto-Holds (A-06)
For orders routed to `REQUIRE_DEPOSIT` (elevated risk):
- The app calls the Shopify Fulfillment Orders API (`fulfillmentOrderHold`).
- Fulfillment is placed on hold with reason: *"Awaiting commitment deposit (₹100)"*.
- An automated SMS/WhatsApp sends a secure UPI payment link for ₹100.
- Upon payment settlement via Razorpay/Cashfree webhook, the hold is released automatically.

---

## 5. Commercial Billing Architecture (A-07)
Integrated with Shopify App Billing API (Recurring Application Charges):
- **Free Bleed Audit:** ₹0/month, up to 100 orders/month (Includes historical CSV bleed report).
- **Starter:** ₹4,999/month, up to 2,500 orders/month, ₹2.00/order overage.
- **Growth:** ₹14,999/month, up to 10,000 orders/month, ₹1.50/order overage (Includes auto-holds, custom margin configuration, and SLA).
