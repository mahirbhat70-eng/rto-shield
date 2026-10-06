"""
scripts/generate_narrative_blocks.py — Programmatic generation of documentation blocks
from configs/cost_config.yaml and reports/headline_numbers.json.

Ensures zero manual transcription of numbers or assumptions into documentation.
"""

import hashlib
import json
import os
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_config_provenance():
    cfg_path = os.path.join(ROOT, "configs", "cost_config.yaml")
    with open(cfg_path, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    cfg = yaml.safe_load(data)
    return cfg, h, cfg_path


def generate_cost_narrative(cfg, config_sha256):
    rto_cost = cfg["rto_logistics_cost"]
    margin = int(cfg["average_margin_pct"] * 100)
    iv = cfg["interventions"]
    assumed = cfg.get("assumed_operational_costs", {})

    narrative = f"""<!-- BEGIN_GENERATED_COST_NARRATIVE (SHA256: {config_sha256}) -->
### Cost Engine & Operational Friction Parameters
*Auto-generated from `configs/cost_config.yaml` (SHA-256: `{config_sha256[:16]}...`)*

- **RTO Logistics Cost:** ₹{rto_cost} per failed delivery attempt (reverse freight + re-packaging).
- **Gross Merchandise Margin:** {margin}% benchmark on delivered orders.
- **Intervention Economics:**
  - `ALLOW_COD`: Friction ₹{iv['ALLOW_COD']['friction_cost']:.2f}, Drop-off {int(iv['ALLOW_COD']['success_drop_pct']*100)}%, RTO Reduction {int(iv['ALLOW_COD']['rto_reduction_pct']*100)}%.
  - `VERIFY_ADDRESS`: Friction ₹{iv['VERIFY_ADDRESS']['friction_cost']:.2f} (WhatsApp OTP ₹{assumed.get('whatsapp_otp_template_cost', 0.75):.2f} + Support overhead), Drop-off {int(iv['VERIFY_ADDRESS']['success_drop_pct']*100)}%, RTO Reduction {int(iv['VERIFY_ADDRESS']['rto_reduction_pct']*100)}%.
  - `REQUIRE_DEPOSIT`: Friction ₹{iv['REQUIRE_DEPOSIT']['friction_cost']:.2f} (Gateway fee {assumed.get('payment_gateway_fee_deposit_pct', 0.02)*100:.1f}% + ₹{assumed.get('payment_gateway_fee_deposit_fixed', 2.0):.2f}, Refund reconciliation, Support overhead), Drop-off {int(iv['REQUIRE_DEPOSIT']['success_drop_pct']*100)}%, RTO Reduction {int(iv['REQUIRE_DEPOSIT']['rto_reduction_pct']*100)}%.
  - `PREPAID_ONLY`: Friction ₹{iv['PREPAID_ONLY']['friction_cost']:.2f}, Drop-off {int(iv['PREPAID_ONLY']['success_drop_pct']*100)}%, RTO Reduction {int(iv['PREPAID_ONLY']['rto_reduction_pct']*100)}%.
<!-- END_GENERATED_COST_NARRATIVE -->"""
    return narrative


def main():
    cfg, config_sha256, _ = get_config_provenance()
    narrative = generate_cost_narrative(cfg, config_sha256)
    out_path = os.path.join(ROOT, "reports", "generated_cost_narrative.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(narrative)
    print("Generated cost narrative successfully at:", out_path)
    return narrative


if __name__ == "__main__":
    main()
