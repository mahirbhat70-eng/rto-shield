# RTO Shield Data Dictionary

This document defines the schema, data types, constraints, and data leakage contract for the RTO Shield synthetic dataset (`data/raw/synthetic_orders.csv`).

Total Columns: **19**  
Dataset Size: **100,000 synthetic order records**

---

## Leakage Prevention & Prediction-Time Contract

> [!IMPORTANT]
> **Prediction-Time Data Contract**:
>
> RTO Shield models merchant risk at the exact moment an order is placed.
>
> - All **18 input features** (columns 1–18) are knowable at order-creation time.
> - `rto_label` (column 19) is **never** available at prediction time and **never** feeds any feature.
> - `historical_pincode_rto_rate` is a synthetic pre-order historical proxy derived from a **pincode-level latent prior** (`θ_pincode ~ Beta(2, 8)`), not computed by grouping or aggregating this dataset's `rto_label` column. The current order's outcome and any future order outcomes are strictly excluded.
> - `prior_orders`, `prior_rto_count`, and `orders_last_24h` are latent per-row attributes sampled independently per (customer, order) rather than reconstructed from the generated 6-month order stream. This is documented explicitly as a **Stage 1 simplification** — it avoids sequential-replay engineering cost while remaining logically consistent (see invariants in §6). Features still represent what a real system would know at order time.
> - The dataset intentionally contains overlap/noise; no feature perfectly separates the classes.

---

## Column Specifications

### A. ORDER ATTRIBUTES (8 Columns)

| Column Name | Data Type | Description | Example Value | Valid Range / Constraints | Role | Available at Order Creation |
|---|---|---|---|---|---|---|
| `order_id` | String | Unique order transaction identifier | `"ORD000000001"` | Unique across all rows | Identifier | Available at Order Creation: YES |
| `timestamp` | Datetime (ISO 8601) | Datetime of order checkout (IST semantics) | `"2026-03-15 14:32:10"` | ~6 months span, sorted ascending | Timestamp | Available at Order Creation: YES |
| `order_value` | Float | Total monetary order value in INR (₹) | `649.50` | Strictly positive, right-skewed (median ≈ ₹600, tail to ~₹15,000) | Feature | Available at Order Creation: YES |
| `quantity` | Integer | Total units ordered | `2` | Integer ≥ 1, capped ~10 | Feature | Available at Order Creation: YES |
| `category` | String | E-commerce product category | `"Electronics"` | `Electronics`, `Apparel`, `Footwear`, `Beauty`, `Home`, `Jewelry` | Feature | Available at Order Creation: YES |
| `discount_pct` | Float | Percentage discount applied at checkout | `12.35` | `0.0 ≤ discount_pct ≤ 70.0`, skewed toward small discounts | Feature | Available at Order Creation: YES |
| `payment_method` | String | Selected payment instrument | `"COD"` | `COD`, `UPI`, `Credit Card`, `Debit Card`, `Net Banking` | Feature | Available at Order Creation: YES |
| `cod_charge` | Float | Merchant's COD service fee in INR (₹) | `49.0` | Exactly `0.0` for non-COD; `[20, 100]` for COD | Feature | Available at Order Creation: YES |

---

### B. CUSTOMER ATTRIBUTES (4 Columns)

| Column Name | Data Type | Description | Example Value | Valid Range / Constraints | Role | Available at Order Creation |
|---|---|---|---|---|---|---|
| `customer_id` | String | Unique customer account identifier | `"CUST000042"` | ~30,000 distinct customers across 100k orders | Identifier | Available at Order Creation: YES |
| `account_age_days` | Integer | Account age in days at checkout | `120` | Integer ≥ 0, cap 3650. If `account_age_days == 0` then `prior_orders` must be `0`. | Feature | Available at Order Creation: YES |
| `prior_orders` | Integer | Count of prior orders attempted/completed | `5` | Integer ≥ 0 | Feature | Available at Order Creation: YES |
| `prior_rto_count` | Integer | Count of prior Return-to-Origin orders | `1` | Integer ≥ 0, MUST satisfy `prior_rto_count ≤ prior_orders` | Feature | Available at Order Creation: YES |

---

### C. DELIVERY / GEOGRAPHY ATTRIBUTES (4 Columns)

| Column Name | Data Type | Description | Example Value | Valid Range / Constraints | Role | Available at Order Creation |
|---|---|---|---|---|---|---|
| `pincode` | String | 6-digit Indian postal code (string, zero-padded) | `"400001"` | Fixed pool of ~1,000 distinct pincodes | Feature | Available at Order Creation: YES |
| `courier_id` | String | Assigned logistics courier partner code | `"Courier_A"` | ~5 distinct couriers | Feature | Available at Order Creation: YES |
| `pincode_tier` | Integer | Geographic tier classification | `1` | Integer in {1, 2, 3}, deterministically mapped from pincode | Feature | Available at Order Creation: YES |
| `historical_pincode_rto_rate` | Float | Historical RTO rate of pincode strictly prior to order | `0.1450` | `0.0 ≤ historical_pincode_rto_rate ≤ 1.0`, mean ∈ [0.10, 0.35] | Feature | Available at Order Creation: YES |

> **Special Oracle-Grade Disclosure for `historical_pincode_rto_rate`**:
> - This feature is **oracle-grade**. It is the exact latent prior (`θ_pincode ~ Beta(2, 8)`) used by the generator to set each row's underlying `P(RTO)`.
> - The per-order feature value = `clip(θ_pincode + N(0, 0.03), 0, 1)`.
> - While strictly backward-looking in the mathematical sense (it contains zero target leakage from future rows), its name overstates what a production system could actually compute. A real merchant must estimate this from finite history (e.g. ±~0.04 noise at ~100 orders/pincode).
> - Because it is the ground-truth risk parameter, it acts as a fidelity advantage in this dataset. It makes every evaluation metric an **optimistic upper bound**. Expect this feature to heavily dominate feature importance.

---

### D. BEHAVIORAL ATTRIBUTES (2 Columns)

| Column Name | Data Type | Description | Example Value | Valid Range / Constraints | Role | Available at Order Creation |
|---|---|---|---|---|---|---|
| `orders_last_24h` | Integer | Count of orders placed by customer in preceding 24h | `2` | Integer ≥ 0, right-skewed (most 0–1) | Feature | Available at Order Creation: YES |
| `device_cluster_size` | Integer | Accounts linked to same device/IP cluster | `1` | Integer ≥ 1, max ~20 (1 at 70%, 2–5 at 25%, 6+ at 5%) | Feature | Available at Order Creation: YES |

---

### E. TARGET VARIABLE (1 Column)

| Column Name | Data Type | Description | Example Value | Valid Range / Constraints | Role | Available at Order Creation |
|---|---|---|---|---|---|---|
| `rto_label` | Integer (Binary) | Ground truth delivery outcome | `1` | Binary: `1` = RTO (Failed/Returned), `0` = Delivered | Target | Available at Order Creation: NO — this is a post-delivery outcome observed only after the courier attempt. It is the supervised-learning target (1 = RTO, 0 = Delivered), not an input feature. |

---

## Feature Summary Table

| Category | Count | Column Names |
|---|---|---|
| **Identifiers** | 2 | `order_id`, `customer_id` |
| **Timestamp** | 1 | `timestamp` |
| **Numerical Features** | 10 | `order_value`, `quantity`, `discount_pct`, `cod_charge`, `account_age_days`, `prior_orders`, `prior_rto_count`, `historical_pincode_rto_rate`, `orders_last_24h`, `device_cluster_size` |
| **Categorical Features** | 5 | `category`, `payment_method`, `pincode`, `courier_id`, `pincode_tier` |
| **Target Variable** | 1 | `rto_label` |
| **Total** | **19** | |

---

## Stage 1 Design Notes (§6)

`prior_orders`, `prior_rto_count`, and `orders_last_24h` are **latent per-row attributes** sampled independently per (customer, order) rather than replayed from the 6-month order stream. This avoids sequential-replay engineering cost. The features still represent what a real production system would know at order creation time and remain logically consistent (invariants: `prior_rto_count ≤ prior_orders`; if `account_age_days == 0` then `prior_orders == 0`). This is a documented Stage 1 simplification.
