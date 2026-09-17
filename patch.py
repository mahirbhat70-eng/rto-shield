import re

with open('dashboard.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Define defaults at the top of the file, right after imports
defaults_code = """
DEFAULTS = {
    "order_value": 852.0, "category": "Home", "payment_method": "COD", "quantity": 3,
    "discount_pct": 19.8, "cod_charge": 59.0, "account_age_days": 65, "prior_orders": 2,
    "prior_rto_count": 0, "orders_last_24h": 3, "device_cluster_size": 1, 
    "pincode": "253407", "courier_id": "Courier_E"
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
"""
text = text.replace('import streamlit as st', 'import streamlit as st\n' + defaults_code)

# Replace the input widgets to remove 'value=' and 'index=' and change keys
# Order Intake
text = re.sub(r'value=float\([^)]+\),\s*key=\"ni_order_value\"', 'key=\"order_value\"', text)
text = re.sub(r'index=CATEGORIES\.index\([^)]+\),\s*key=\"sb_category\"', 'key=\"category\"', text)
text = re.sub(r'index=\[\"COD\", \"PREPAID\"\]\.index\([^)]+\),\s*key=\"sb_payment_method\"', 'key=\"payment_method\"', text)
text = re.sub(r'value=int\([^)]+\),\s*key=\"ni_quantity\"', 'key=\"quantity\"', text)
text = re.sub(r'value=float\([^)]+\),\s*key=\"ni_discount_pct\"', 'key=\"discount_pct\"', text)

# COD charge
text = re.sub(r'value=float\([^)]+\),\s*key=\"ni_cod_charge\",', 'key=\"cod_charge\",', text)

# Customer History
text = re.sub(r'value=int\([^)]+\),\s*key=\"ni_account_age_days\"', 'key=\"account_age_days\"', text)
text = re.sub(r'value=int\([^)]+\),\s*key=\"ni_prior_orders\"', 'key=\"prior_orders\"', text)
text = re.sub(r'value=min\([^)]+\)\),\s*key=\"sl_prior_rto_count\"', 'key=\"prior_rto_count\"', text)

# Velocity & Logistics
text = re.sub(r'value=int\([^)]+\),\s*key=\"ni_orders_last_24h\"', 'key=\"orders_last_24h\"', text)
text = re.sub(r'value=int\([^)]+\),\s*key=\"ni_device_cluster_size\"', 'key=\"device_cluster_size\"', text)

# Pincode & Courier
text = re.sub(r'value=str\([^)]+\),\s*key=\"ti_pincode\"', 'key=\"pincode\"', text)
text = re.sub(r'value=str\([^)]+\),\s*key=\"ti_courier_id\"', 'key=\"courier_id\"', text)

# Payload from state
text = text.replace('st.session_state["sb_payment_method"]', 'st.session_state["payment_method"]')
text = text.replace('st.session_state["ni_order_value"]', 'st.session_state["order_value"]')
text = text.replace('st.session_state["sb_category"]', 'st.session_state["category"]')
text = text.replace('st.session_state["ni_quantity"]', 'st.session_state["quantity"]')
text = text.replace('st.session_state["ni_discount_pct"]', 'st.session_state["discount_pct"]')
text = text.replace('st.session_state["ni_cod_charge"]', 'st.session_state["cod_charge"]')
text = text.replace('st.session_state["ni_account_age_days"]', 'st.session_state["account_age_days"]')
text = text.replace('st.session_state["ni_prior_orders"]', 'st.session_state["prior_orders"]')
text = text.replace('st.session_state["sl_prior_rto_count"]', 'st.session_state["prior_rto_count"]')
text = text.replace('st.session_state["ni_orders_last_24h"]', 'st.session_state["orders_last_24h"]')
text = text.replace('st.session_state["ni_device_cluster_size"]', 'st.session_state["device_cluster_size"]')
text = text.replace('st.session_state["ti_pincode"]', 'st.session_state["pincode"]')
text = text.replace('st.session_state["ti_courier_id"]', 'st.session_state["courier_id"]')

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patched dashboard.py")
