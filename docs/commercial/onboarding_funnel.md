# FlipPrice AI — Merchant Onboarding Funnel & Telemetry (A-09, A-11)

## 1. The 3-Step "Time to Value" Setup Funnel

```
[Step 1: 1-Click Install] ──> [Step 2: 10-Minute Bleed Report] ──> [Step 3: Enable Auto-Tagging]
      (OAuth & Scopes)              (Upload Historical CSV)             (Switch on Live Pricing)
```

### Step 1: Connect Store (Target: < 45 seconds)
* Merchant installs app from Shopify App Store.
* OAuth completes with zero custom developer configuration.
* Tenant is registered in SQLite with an isolated API key (`fp_live_shopify_...`).

### Step 2: Historical Bleed Report (The "Hook") (Target: < 10 minutes)
* Merchant uploads past 90-day Shopify order export CSV.
* `bleed_report.py` executes in < 30 seconds.
* Displays the ₹ Bleed Breakdown:
  - Total historical RTO loss: e.g. ₹3,42,000.
  - Losses caused by static blocklist: ₹1,14,000 in honest customer margin forfeited.
  - Net savings FlipPrice would have captured: ₹88,400.
* Conversion trigger: *"Unlock these savings on your live orders starting today."*

### Step 3: Enable Live Auto-Tagging (Target: < 1 minute)
* Merchant toggles `Enable Live Order Tagging`.
* Webhook subscription (`orders/create`) activates.
* Every subsequent order receives tags (`fp_risk:`, `fp_action:`, `fp_el:`).

---

## 2. Funnel Conversion Telemetry & Drop-Off Mitigation (A-11)

| Funnel Transition | Benchmark Goal | Instrumentation Event | Automated Mitigation |
| :--- | :--- | :--- | :--- |
| Install $\rightarrow$ First Bleed Report | 85% | `app_installed`, `csv_uploaded` | Trigger automated email with sample audit video if no upload within 2 hours. |
| Bleed Report $\rightarrow$ Live Tagging Active | 70% | `report_viewed`, `tagging_enabled` | Show exact rupee difference between merchant's current RTO rate and FlipPrice post-policy rate. |
| Trial (14d) $\rightarrow$ Paid Subscriber | 65% | `trial_started`, `plan_subscribed` | Display weekly "Rupees Saved by FlipPrice" notification inside Shopify Admin. |
