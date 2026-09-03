# Stage 4 Financial Results (COD Subset Only)

**Tuned Thresholds (val_cal)**:
- PREPAID_ONLY Threshold: 0.48
- VERIFY_ADDRESS Threshold: 0.20

### 1. Always Allow (Baseline)
- **Total Loss (INR)**: -285733.04
- **Total Margin Saved**: 0.00
- **Delta Savings (Noise Test)**: 0.00
- **Operational Counts**:
  - Orders Touched: 0
  - Expected RTOs Prevented: 0.00
  - Expected Good-Customer Drops: 0.00
  - Friction Spend: 0.00
- **Action Distribution & P-Bands**:
  - ALLOW_COD: 100.0% (P-band: 0.1509 - 1.0000, median 0.2745)

### 2. Binary PREPAID Block
- **Total Loss (INR)**: -286285.78
- **Total Margin Saved**: 552.74
- **Delta Savings (Noise Test)**: -669.11
- **Operational Counts**:
  - Orders Touched: 36
  - Expected RTOs Prevented: 12.07
  - Expected Good-Customer Drops: 9.83
  - Friction Spend: 0.00
- **Action Distribution & P-Bands**:
  - ALLOW_COD: 99.0% (P-band: 0.1509 - 0.4754, median 0.2745)
  - PREPAID_ONLY: 1.0% (P-band: 0.5000 - 1.0000, median 0.6000)

### 3. Binary VERIFY Block
- **Total Loss (INR)**: -303727.32
- **Total Margin Saved**: 17994.28
- **Delta Savings (Noise Test)**: 470.88
- **Operational Counts**:
  - Orders Touched: 3027
  - Expected RTOs Prevented: 277.28
  - Expected Good-Customer Drops: 105.14
  - Friction Spend: 6054.00
- **Action Distribution & P-Bands**:
  - VERIFY_ADDRESS: 83.0% (P-band: 0.2131 - 1.0000, median 0.2922)
  - ALLOW_COD: 17.0% (P-band: 0.1509 - 0.1958, median 0.1720)

### 4. Multi-Action AI Policy (Primary)
- **Total Loss (INR)**: -322053.05
- **Total Margin Saved**: 36320.01
- **Delta Savings (Noise Test)**: 1173.39
- **Operational Counts**:
  - Orders Touched: 2952
  - Expected RTOs Prevented: 482.99
  - Expected Good-Customer Drops: 422.10
  - Friction Spend: 3206.00
- **Action Distribution & P-Bands**:
  - VERIFY_ADDRESS: 44.0% (P-band: 0.1509 - 0.6000, median 0.2475)
  - REQUIRE_DEPOSIT: 37.0% (P-band: 0.1509 - 1.0000, median 0.3312)
  - ALLOW_COD: 19.0% (P-band: 0.1509 - 0.4754, median 0.2250)

### 5. Multi-Action AI Policy (Sensitivity)
- **Total Loss (INR)**: -327032.94
- **Total Margin Saved**: 41299.90
- **Delta Savings (Noise Test)**: -447.63
- **Operational Counts**:
  - Orders Touched: 2941
  - Expected RTOs Prevented: 456.22
  - Expected Good-Customer Drops: 414.39
  - Friction Spend: 3306.00
- **Action Distribution & P-Bands**:
  - VERIFY_ADDRESS: 45.3% (P-band: 0.1314 - 0.5837, median 0.2616)
  - REQUIRE_DEPOSIT: 35.3% (P-band: 0.1362 - 0.6935, median 0.2999)
  - ALLOW_COD: 19.3% (P-band: 0.1303 - 0.4547, median 0.2161)

### 6. Multi-Action AI Policy (Constraint-Clipped Sensitivity)
- **Total Loss (INR)**: -322074.02
- **Total Margin Saved**: 36340.98
- **Delta Savings (Noise Test)**: 1173.05
- **Operational Counts**:
  - Orders Touched: 2952
  - Expected RTOs Prevented: 482.87
  - Expected Good-Customer Drops: 422.16
  - Friction Spend: 3206.00
- **Action Distribution & P-Bands**:
  - VERIFY_ADDRESS: 44.0% (P-band: 0.1509 - 0.6000, median 0.2475)
  - REQUIRE_DEPOSIT: 37.0% (P-band: 0.1509 - 0.8500, median 0.3312)
  - ALLOW_COD: 19.0% (P-band: 0.1509 - 0.4754, median 0.2250)

