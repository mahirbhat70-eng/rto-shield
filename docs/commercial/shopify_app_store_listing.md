# FlipPrice AI — Shopify App Store Listing & Compliance Packet

## App Name & Header
* **App Name:** FlipPrice AI: RTO & COD Pricing
* **Tagline:** Blocklists guess. We price. Cut RTO losses without forfeiting sales.
* **Category:** Orders and shipping > Fraud prevention & Return optimization
* **Pricing:** Free to install. Paid plans from ₹4,999/month. Free 90-day historical bleed audit included.

---

## 1. Key Value Propositions (Merchant Bullets)
1. **Stop Rejecting Real Buyers:** Static pincode blocklists forfeit 15–25% of genuine orders. FlipPrice prices risk per order so you keep the customer.
2. **Cheapest Safe Action per Order:** Automatically routes borderline orders to low-friction WhatsApp verification (₹2) and reserves deposit requests (₹100) for extreme risks.
3. **Frictionless Zero-Theme Install:** Works out-of-the-box via native Shopify Order Tags and Metafields. No theme code modifications, no checkout breaking changes.
4. **Automated WhatsApp Confirmation:** One-click confirmation buttons (Confirm Order / Cancel) via Gupshup/Wati/Interakt templates before courier dispatch.
5. **Instant Fulfillment Auto-Hold:** Automatically pauses fulfillment on deposit-pending orders to protect warehouse pick-and-pack expenses.

---

## 2. Keywords & Search Optimization
* **Primary Keywords:** RTO reduction, Cash on Delivery, COD verification, fake order filter, pincode blocklist alternative, return to origin, Shopify India COD, WhatsApp order confirmation, order tagging.

---

## 3. Protected Customer Data Compliance (A-13)
* **DPDP Act 2023 & Shopify Data Protection:**
  - Customer phone numbers and email addresses are salted and hashed using SHA-256 (`hash_pii`) upon ingestion.
  - Raw PII is never stored in server logs or database tables.
  - Mandatory Shopify Webhook endpoints implemented:
    - `customers/data_request` $\rightarrow$ Returns cryptographic hash log.
    - `customers/redact` $\rightarrow$ Purges associated customer telemetry.
    - `shop/redact` $\rightarrow$ Cleanses store tenant context within 48 hours.

---

## 4. App Store Review Submission Tracker (A-14)
| Milestone | Status | Target Date | Owner |
| :--- | :--- | :--- | :--- |
| Core Webhook & Decision Engine Tests Green | ✅ Verified (285+ tests) | 2026-10-08 | Squad S5 |
| OAuth & Session Persistence on SQLite | ✅ Complete | 2026-10-08 | Squad S5 |
| Adversarial & Security Red-Team Sign-Off | ✅ Complete | 2026-10-08 | Squad S9 |
| Dev Store Sandbox Test & Tag Verification | ✅ Verified | 2026-10-08 | Gate 3 |
| Shopify App Review Submission | ⏳ Pending Partner Portal Push | 2026-10-10 | Founder |
