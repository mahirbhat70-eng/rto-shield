import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import time
import json
import datetime
from src.serve.scorer import score_order

# Page config
st.set_page_config(page_title="Order Risk Scorer", layout="wide")

@st.cache_resource
def load_scorer_assets():
    # The module-level imports in scorer.py already loaded models,
    # but this ensures streamlit registers the cache hit and boots fast.
    return True

load_scorer_assets()

# Preset Demo Configurations
PRESETS = {
    "DEPOSIT": {
        "order_value": 186.0, "category": "Beauty", "payment_method": "COD", "quantity": 1,
        "discount_pct": 0.0, "cod_charge": 29.0, "account_age_days": 260, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 1, "device_cluster_size": 1, 
        "pincode": "461780", "courier_id": "Courier_A"
    },
    "ALLOW": {
        "order_value": 1344.0, "category": "Electronics", "payment_method": "COD", "quantity": 4,
        "discount_pct": 0.0, "cod_charge": 49.0, "account_age_days": 998, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 2, "device_cluster_size": 4, 
        "pincode": "750176", "courier_id": "Courier_B"
    },
    "VERIFY": {
        "order_value": 852.0, "category": "Home", "payment_method": "COD", "quantity": 3,
        "discount_pct": 19.8, "cod_charge": 59.0, "account_age_days": 65, "prior_orders": 2,
        "prior_rto_count": 0, "orders_last_24h": 3, "device_cluster_size": 1, 
        "pincode": "253407", "courier_id": "Courier_E"
    }
}

def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

st.sidebar.markdown("### Demo Presets")
c1, c2, c3 = st.sidebar.columns(3)
c1.button("DEPOSIT", on_click=load_preset, args=("DEPOSIT",), help="₹186, 21.31%")
c2.button("ALLOW", on_click=load_preset, args=("ALLOW",), help="₹1344, 21.31%")
c3.button("VERIFY", on_click=load_preset, args=("VERIFY",), help="₹852, 39.45%")
st.sidebar.markdown("---")

st.sidebar.title("Order Inputs")

# Helper to get session state or default
def get_val(key, default):
    return st.session_state.get(key, default)

order_value = st.sidebar.number_input("Order Value (₹)", min_value=0.0, value=get_val("order_value", 616.0), step=10.0, key="ni_order_value")
category = st.sidebar.selectbox("Category", ['Apparel', 'Home', 'Electronics', 'Beauty', 'Footwear', 'Jewelry'], index=['Apparel', 'Home', 'Electronics', 'Beauty', 'Footwear', 'Jewelry'].index(get_val("category", "Apparel")), key="sb_category")
payment_method = st.sidebar.selectbox("Payment Method", ['COD', 'PREPAID'], index=['COD', 'PREPAID'].index(get_val("payment_method", "COD")), key="sb_payment_method")
quantity = st.sidebar.number_input("Quantity", min_value=1, value=get_val("quantity", 2), key="ni_quantity")
discount_pct = st.sidebar.number_input("Discount %", min_value=0.0, max_value=100.0, value=get_val("discount_pct", 2.0), key="ni_discount_pct")

if payment_method == 'COD':
    cod_charge = st.sidebar.number_input("COD Charge (₹)", min_value=0.0, value=get_val("cod_charge", 49.0), help="Train COD median ₹49", key="ni_cod_charge")
else:
    cod_charge = st.sidebar.number_input("COD Charge (₹)", min_value=0.0, max_value=0.0, value=0.0, disabled=True, key="ni_cod_charge_disabled")

account_age_days = st.sidebar.number_input("Account Age (Days)", min_value=0, value=get_val("account_age_days", 236), key="ni_account_age_days")
prior_orders = st.sidebar.number_input("Prior Orders", min_value=0, value=get_val("prior_orders", 3), key="ni_prior_orders")
prior_rto_count = st.sidebar.slider("Prior RTO Count", min_value=0, max_value=max(1, prior_orders), value=get_val("prior_rto_count", 0), key="sl_prior_rto_count")
if prior_rto_count > prior_orders:
    st.sidebar.warning("Prior RTOs cannot exceed prior orders.")
    
orders_last_24h = st.sidebar.number_input("Orders Last 24h", min_value=0, value=get_val("orders_last_24h", 1), key="ni_orders_last_24h")
device_cluster_size = st.sidebar.number_input("Device Cluster Size", min_value=1, value=get_val("device_cluster_size", 1), key="ni_device_cluster_size")
# 597542 is a known valid pincode in the lookup
pincode = st.sidebar.text_input("Pincode", value=get_val("pincode", "597542"), key="ti_pincode")
courier_id = st.sidebar.text_input("Courier ID", value=get_val("courier_id", "Courier_A"), key="ti_courier_id")

st.title("Order Risk Scorer")

if st.button("Score this order", type="primary"):
    payload = {
        'order_value': float(order_value),
        'category': category,
        'payment_method': payment_method,
        'quantity': int(quantity),
        'discount_pct': float(discount_pct),
        'cod_charge': float(cod_charge),
        'account_age_days': int(account_age_days),
        'prior_orders': int(prior_orders),
        'prior_rto_count': int(prior_rto_count),
        'orders_last_24h': int(orders_last_24h),
        'device_cluster_size': int(device_cluster_size),
        'pincode': str(pincode),
        'courier_id': str(courier_id)
    }
    
    start_time = time.time()
    try:
        res = score_order(payload)
        st.caption(f"Scored in {time.time() - start_time:.3f}s")
        
        st.header(f"P(RTO): {res['probability']*100:.1f}%")
        
        if payment_method == 'PREPAID':
            st.info("PREPAID -> PASSTHROUGH (No risk of COD RTO)")
            st.success(f"Recommended Action: **{res['recommended_action']}**")
        else:
            st.success(f"Recommended Action: **{res['recommended_action']}**")
            st.write("Expected Loss Table (Argmin Highlighted)")
            el_df = pd.DataFrame([res['el_table']])
            
            # Highlight the column with the minimum value
            min_col = el_df.idxmin(axis=1).iloc[0]
            
            def highlight_min(s):
                return ['background-color: #2e7d32; color: white' if col == min_col else '' for col in s.index]
                
            st.dataframe(el_df.style.apply(highlight_min, axis=1))
            
        st.subheader("Top Risk Factors")
        for name, shap_val in res['shap_top_factors']:
            direction = "↑ increases risk" if shap_val > 0 else "↓ reduces risk"
            st.write(f"- **{name}**: {shap_val:+.4f} ({direction})")

        # Record into audit trail
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload": payload,
            "probability": round(float(res['probability']), 4),
            "recommended_action": res['recommended_action'],
            "expected_losses": {k: round(float(v), 2) for k, v in res['el_table'].items()} if 'el_table' in res else {},
            "top_factors": [{"feature": name, "shap_val": round(float(val), 4)} for name, val in res['shap_top_factors']]
        }
        st.session_state.setdefault("audit_log", []).append(entry)
            
    except ValueError as e:
        msg = str(e)
        if "exceeds maximum allowed value" in msg:
            st.error("Input outside tested range (guardrail rejected).")
            st.caption(msg)
        else:
            st.error(msg)

# Audit Trail Export
if st.session_state.get("audit_log"):
    st.markdown("---")
    st.subheader(f"📋 Decision Audit Trail ({len(st.session_state['audit_log'])} logged)")
    jsonl_str = "\n".join(json.dumps(item) for item in st.session_state["audit_log"])
    col_dl, col_clr = st.columns([3, 1])
    with col_dl:
        st.download_button(
            label="📥 Download Audit Trail (.jsonl)",
            data=jsonl_str,
            file_name="rto_audit_trail.jsonl",
            mime="application/x-ndjson",
            use_container_width=True
        )
    with col_clr:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state["audit_log"] = []
            st.rerun()
            
st.markdown("---")
st.caption("Frozen v1.0 artifacts · calibrated LightGBM · expected-loss argmin · all numbers reproducible via reports/")
