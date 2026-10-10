# Brand Strategy & Commercial Identity: FlipPrice AI

> **Status:** Phase 0.1 Complete  
> **Primary Selection:** **FlipPrice AI** (Engine: `rto-shield`)  
> **Backup 1:** **Codewise AI**  
> **Backup 2:** **ShipSure AI**  

---

## 1. The Naming Collision Problem

In the Indian e-commerce & logistics landscape, the name **"RTO Shield"** is hopelessly saturated:
1. **RTO Shield – Reduce fake orders** (KLIP SOLUTIONS): A free Shopify app live since July 2024. Uses static pincode blocklists, has a 0.0 rating and 0 reviews after 2+ years.
2. **RTO Shield** (Adoment): A closed-box WhatsApp confirmation and auto-hold platform.
3. **rtoshield.in**: Multi-carrier shipping software (domain collision).
4. **TrueReach™ RTO Shield** (JetPost): Post-purchase NDR recovery service.

### The Strategy: Separate Engine from Brand
* The open-source and audited decision engine remains **`rto-shield`**.
* The merchant-facing commercial AI SaaS is branded as **FlipPrice AI**.

---

## 2. Name Candidate Evaluation & Screening Matrix

We evaluated 12 candidates against four strict criteria:
- **Semantic fit:** Expresses decision-pricing over blunt blocking.
- **Shopify App Store search:** Zero collision with live apps.
- **Domain viability:** `.ai`, `.in`, or `.com` availability/acquisition tier.
- **India Trademark (Classes 9 & 42):** Clean search in IP India public database.

| Candidate | Positioning Concept | Shopify Collision | Trademark Risk (Class 9/42) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **1. FlipPrice AI** | *"Don't predict the coin flip. Price it."* | **None** | Low / Distinctive | **Winner (Primary)** |
| **2. Codewise AI** | *"Smarter COD decisions, rupee-by-rupee"* | Low (generic apps exist, no RTO) | Medium (software classes) | **Backup 1** |
| **3. ShipSure AI** | *"Certainty on every dispatch"* | Low (ShipSure logistics exists) | Medium | **Backup 2** |
| **4. TrueOrder** | *"Verified intent, zero friction"* | High ("TrueOrder" POS apps) | High | Dropped |
| **5. BeforeShip** | *"Pre-dispatch risk arbitrage"* | None | Low | Viable, but sounds like workflow tool |
| **6. MarginGuard** | *"Protect the ₹165 margin on every order"* | Medium (Shopify margin calculators) | Low | Over-indexed on finance |
| **7. ClearCOD** | *"Transparent COD pricing & recovery"* | Medium | Low | Sounds like a payment gateway |
| **8. AntiBleed** | *"Stop bleeding lakhs on COD returns"* | None | Low | Too aggressive/negative |
| **9. ReturnPrice** | *"Price return probability in rupees"* | None | Low | Functional but dry |
| **10. DeRisk COD** | *"Four-way order risk routing"* | High | Medium | Generic descriptor |
| **11. CartShield** | *"Protection at checkout"* | High | High | Too close to checkout apps |
| **12. Argmin AI** | *"Minimum expected loss routing"* | None | Low | Too academic for D2C founders |

---

## 3. Winning Brand Identity: FlipPrice AI

### Core Positioning
> **Tagline:** Blocklists guess. We price.  
> **One-Liner:** FlipPrice is the per-order decision engine that replaces blunt COD blocklists with four calibrated actions—saving Indian D2C brands up to ₹55,000+ per 7,000 orders in measurable net margin.

### The Problem It Solves
Traditional RTO tools block entire pincodes or phones when risk is elevated. But blocking a pincode to stop one bad order forfeits ₹165 in net product margin on honest customers. Static blacklists lose up to **₹3.44 lakh** on 7,174 orders unless 68% of blocked customers convert to prepaid (which never happens in India).

FlipPrice calculates calibrated return probabilities down to **ECE 0.0045** and selects the action with the **minimum expected loss (argmin)**:
1. **ALLOW_COD**: Low risk ($p < 0.21$), preserve maximum conversion.
2. **VERIFY_ADDRESS**: Moderate risk, trigger automated ₹7 WhatsApp verification.
3. **REQUIRE_DEPOSIT**: High risk, collect ₹50 refundable advance deposit before fulfillment.
4. **FORCE_PREPAID**: Extreme risk ($p > 0.85$), disable COD completely.

### Target Persona (ICP)
* **Indian D2C Merchants:** Monthly GMV between ₹10L and ₹1Cr.
* **COD Volume:** 25% to 60% cash-on-delivery orders.
* **Ecosystem:** Shopify storefront + Shiprocket / Delhivery courier orchestration.
* **Pain Point:** Bleeding 25–40% RTO rate on COD orders while losing honest customers to rigid blacklist apps.

---

## 4. Legal & Regulatory Screen (India IP & DPDP Act 2023)

### 4.1 Trademark Screen
* **Class 9 (Software & Computer Programs):** "FlipPrice" is unregistered and phonetically novel.
* **Class 42 (Software as a Service / Cloud Services):** Clean search. Filing priority recommended upon incorporation.

### 4.2 DPDP Act 2023 Architecture Requirements
FlipPrice processes phone numbers, delivery addresses, and customer order histories. To comply with India's Digital Personal Data Protection Act (DPDP) 2023:
1. **PII Ingestion Hashing:** Phone numbers and email addresses are salted and hashed (`SHA-256 + secret salt`) upon ingestion. Unmasked PII is never stored in model feature tables.
2. **Pincode Aggregation:** Pincode risk signals rely on postal division aggregations, never individual household addresses.
3. **Right to Erasure (Offboarding):** Tenants can purge historical transaction logs and audit records via a one-click API deletion webhook.
4. **Zero Third-Party Model Training:** Merchant data is strictly isolated by tenant ID and never shared across merchants or used to train global external LLMs.

---

## 5. Visual System & Product UI Tokens

* **Primary Dark Background:** `#0B0F19` (Deep slate / navy)
* **Card & Surface Background:** `#131B2E` with 1px border `#1E293B`
* **Brand Accent:** `#38BDF8` (Sky Cyan - precision, speed)
* **Success (ALLOW):** `#10B981` (Emerald)
* **Warning (VERIFY):** `#F59E0B` (Amber)
* **Intervention (DEPOSIT):** `#8B5CF6` (Violet)
* **Danger (FORCE_PREPAID):** `#EF4444` (Crimson)
* **Typography:** Inter / Outfit (Modern tabular numerals for financial figures)
