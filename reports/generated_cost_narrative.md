<!-- BEGIN_GENERATED_COST_NARRATIVE (SHA256: 28f9d9317689520d86be28ace066230ab553af6260817d7dfc978461b708c1e2) -->
### Cost Engine & Operational Friction Parameters
*Auto-generated from `configs/cost_config.yaml` (SHA-256: `28f9d9317689520d...`)*

- **RTO Logistics Cost:** ₹150 per failed delivery attempt (reverse freight + re-packaging).
- **Gross Merchandise Margin:** 20% benchmark on delivered orders.
- **Intervention Economics:**
  - `ALLOW_COD`: Friction ₹0.00, Drop-off 0%, RTO Reduction 0%.
  - `VERIFY_ADDRESS`: Friction ₹2.00 (WhatsApp OTP ₹0.75 + Support overhead), Drop-off 5%, RTO Reduction 30%.
  - `REQUIRE_DEPOSIT`: Friction ₹7.00 (Gateway fee 2.0% + ₹1.80, Refund reconciliation, Support overhead), Drop-off 40%, RTO Reduction 80%.
  - `PREPAID_ONLY`: Friction ₹0.00, Drop-off 70%, RTO Reduction 55%.
<!-- END_GENERATED_COST_NARRATIVE -->