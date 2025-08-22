import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import random
import joblib

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk Insights Dashboard")

# --- Dataset selector ---
dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    [
        "Tracking of users",
        "Probability of default(%)"
    ]
)

model_map = {
    "Tracking of users": "credit_risk_model_reg.pkl",
    "Probability of default(%)": "credit_model.pkl"
}

# --- Load dataset ---
if dataset_choice == "Tracking of users.xlsx":
    df = pd.read_excel("Tracking of users.xlsx")
else:
    df = pd.read_excel("loan_repayment_status_final_v3_cleaned.xlsx")

# --- Normalize column names for consistency ---
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Normalize target name
if "loan_repayment_status" not in df.columns:
    loan_stat_col = next((c for c in df.columns if c.replace("_", "") == "loanrepaymentstatus"), None)
    if loan_stat_col:
        df = df.rename(columns={loan_stat_col: "loan_repayment_status"})

# --- Load model ---
if dataset_choice == "Loan Repayment (Custom Dataset)":
    model = joblib.load("credit_model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    try:
        saved = joblib.load(model_map[dataset_choice])
        model = saved["model"]
        model_features = saved.get("features", None)
    except FileNotFoundError:
        st.error(f"Model file {model_map[dataset_choice]} not found.")
        st.stop()
    try:
        saved = joblib.load(model_map[dataset_choice])
    # Check if saved is a dict before accessing ["model"]
        if isinstance(saved, dict):
            model = saved["model"]
            model_features = saved.get("features", None)
        else:
            model = saved  # saved is already the model object
            model_features = None  # or set appropriately if you have features stored elsewhere
    except FileNotFoundError:
        st.error(f"Model file {model_map[dataset_choice]} not found.")
        st.stop()


# --- Sidebar: select user or manual ---
st.sidebar.title("User selection")
user_id_cols = [c for c in df.columns if c in ("user_id", "user id")]
user_ids = df[user_id_cols[0]].tolist() if user_id_cols else []
selected_user = st.sidebar.selectbox("Choose User ID", ["Manual Input"] + user_ids)

if selected_user != "Manual Input" and user_id_cols:
    id_col = user_id_cols[0]
    if df[id_col].dtype == object:
        user_row = df[df[id_col] == selected_user].iloc[0]
    else:
        user_row = df[df[id_col] == type(df[id_col].iloc[0])(selected_user)].iloc
    user_idx = df[df[id_col] == selected_user].index[0]
else:
    user_row = None
    user_idx = None

# --- Base features (exclude target & user id) ---
exclude = user_id_cols + ["loan_repayment_status", "home_ownership"]
base_feature_cols = [c for c in df.columns if c not in exclude]

st.subheader("Applicant features")
input_vals = {}

for col in base_feature_cols:
    norm_col = col.lower().replace(" ", "_")
    if df[col].dtype == "object":
        options = df[col].unique().tolist()
        if user_row is not None:
            default = user_row[col]
        else:
            default = options[0]
        input_vals[norm_col] = st.selectbox(col, options, index=options.index(default))
    else:
        min_v, max_v = float(df[col].min()), float(df[col].max())
        if user_row is not None:
            default = user_row[col]
        else:
            default = float(df[col].mean())
        safe_default = round(float(default), 2)
        step = 1.0 if float(default).is_integer() else 0.01
        input_vals[norm_col] = st.number_input(
            col, value=safe_default,
            min_value=round(min_v, 2),
            max_value=round(max_v, 2),
            step=step
        )

# --- Tracking feature calculation (RANDOM for each user/manual) ---
def compute_tracking_random(user_idx=None):
    if user_idx is not None:
        random.seed(int(user_idx))  # Fix here!
    else:
        random.seed()
    vals = {}
    vals["Debt-to-Income Ratio"] = round(random.uniform(0.1, 0.7), 4)
    vals["Credit Utilization Rate"] = round(random.uniform(0.05, 1.0), 4)
    vals["On-Time Payment Ratio"] = round(random.uniform(0.5, 1.0), 4)
    vals["Delinquency Rate"] = round(random.uniform(0.0, 0.5), 4)
    # Removed Account Age (months)
    return vals

tracking_vals = compute_tracking_random(user_idx)
final_input = {**input_vals, **tracking_vals}
# --- Prediction data prep ---
if dataset_choice == "Loan Repayment (Custom Dataset)":
    # Extract only numeric features from final_input
    numeric_features = [col for col, val in final_input.items() if isinstance(val, (int, float))]
    
    df_numeric = pd.DataFrame([{k: final_input[k] for k in numeric_features}])
    
    try:
        scaler_data = joblib.load("scaler.pkl")
        model_features = scaler_data.get("features", df_numeric.columns.tolist())
        df_numeric = df_numeric[model_features]  # ensure correct order
        df_numeric_scaled = scaler_data["scaler"].transform(df_numeric)
    except:
        df_numeric_scaled = scaler.transform(df_numeric)
    
    # Replace scaled values back into final_input
    for i, col in enumerate(df_numeric.columns):
        final_input[col] = df_numeric_scaled[0][i]
    
    # Full input for prediction (numeric + categorical)
    Xdf = pd.DataFrame([final_input])
else:
    if 'model_features' in locals() and model_features:
        X_row = {}
        for feat in model_features:
            match = next((k for k in final_input.keys()
                          if k.lower().replace(" ", "_") == feat.lower().replace(" ", "_")), None)
            X_row[feat] = final_input[match] if match else 0.0
        Xdf = pd.DataFrame([X_row])
    else:
        Xdf = pd.DataFrame([final_input])

#added new if want remove
if "active_loans_count" in Xdf.columns:
    Xdf["active_loans_count"] = Xdf["active_loans_count"].astype(int)

if "phone_model" in Xdf.columns:
    # If phone_model is numeric but should be int
    Xdf["phone_model"] = Xdf["phone_model"].astype(int)

for c in Xdf.select_dtypes(include=["object"]).columns:
    Xdf[c] = pd.factorize(Xdf[c])[0]

pred = model.predict(Xdf)
try:
    pred_proba = model.predict_proba(Xdf)[1] * 100
except:
    pred_proba = pred * 100

st.header("Credit Risk Prediction")
if pred_proba < 40:
    risk_level, color = "Low Risk", "green"
elif pred_proba < 70:
    risk_level, color = "Medium Risk", "orange"
else:
    risk_level, color = "High Risk", "red"

st.markdown(
    f"""
    <div style='font-size:18px; font-weight:bold;'>Predicted probability of repayment</div>
    <div style='font-size:30px; font-weight:bold; color:{color};'>{float(pred_proba):.2f}%</div>
    <div style='font-size:16px; color:{color};'>{risk_level}</div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar macroeconomic factors ---
st.sidebar.markdown("### Macroeconomic Factors")
for macro in ["gdp_growth", "gdp growth", "GDP Growth (%)", "GDP Growth"]:
    norm_macro = macro.lower().replace(" ", "_")
    if norm_macro in final_input:
        st.sidebar.metric("GDP Growth", final_input[norm_macro])
        break
for m in ["inflation", "Inflation (%)", "Inflation"]:
    norm_inf = m.lower().replace(" ", "_")
    if norm_inf in final_input:
        st.sidebar.metric("Inflation", final_input[norm_inf])
        break
for m in ["unemployment_rate", "Unemployment Rate (%)", "Unemployment Rate"]:
    norm_unemp = m.lower().replace(" ", "_")
    if norm_unemp in final_input:
        st.sidebar.metric("Unemployment Rate", final_input[norm_unemp])
        break

# --- Sidebar tracking indicators ---
st.sidebar.markdown("### Tracking Indicators")
for k, v in tracking_vals.items():
    st.sidebar.metric(k, v)

# --- SHAP Explainability ---
st.header("Explainability (SHAP)")
try:
    explainer = shap.Explainer(model, Xdf)
    shap_values = explainer(Xdf)

    st.subheader("Waterfall plot")
    fig_w, ax_w = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig_w, bbox_inches="tight")

    st.subheader("Summary plot")
    fig_s, ax_s = plt.subplots()
    shap.summary_plot(shap_values, Xdf, show=False)
    st.pyplot(fig_s, bbox_inches="tight")

except Exception as e:
    st.write("SHAP explanation not available:", e)
