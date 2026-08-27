import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from risk_engine import (
    build_chargeback_evidence,
    calculate_abuse_ring_risk,
    calculate_chargeback_risk,
    calculate_return_risk,
    calculate_risk,
    detect_fraud_spike,
)

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
)

HISTORY_DIR = ROOT_DIR / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "transactions_history.csv"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_COLUMNS = [
    "timestamp", "amount", "amount_deviation", "payment_method",
    "merchant_category", "transactions_last_10min", "transactions_last_1h",
    "transactions_last_24h", "device_age_days", "device_transaction_count",
    "location_distance_km", "ip_risk_score", "failed_attempts_24h",
    "previous_fraud_count", "customer_age_days", "customer_avg_amount",
    "account_velocity_score", "chargeback_history", "transaction_hour",
    "is_weekend", "order_count", "return_count", "rto_count",
    "failed_delivery_count", "cod_orders", "linked_accounts",
    "shared_device_accounts", "shared_ip_accounts", "fraud_probability",
    "risk_score", "risk_level", "recommended_action", "return_risk_score",
    "chargeback_risk_score", "abuse_ring_risk_score",
]


def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        df = pd.read_csv(HISTORY_FILE)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col.endswith("_score") else None
        return df
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_transaction(tx, fraud, ret, cb, abuse):
    row = {
        **tx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fraud_probability": fraud["fraud_probability"],
        "risk_score": fraud["risk_score"],
        "risk_level": fraud["risk_level"],
        "recommended_action": fraud["recommended_action"],
        "return_risk_score": ret["score"],
        "chargeback_risk_score": cb["score"],
        "abuse_ring_risk_score": abuse["score"],
    }
    old = load_history()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
        HISTORY_FILE, index=False
    )


def level(score):
    return "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"


st.title("🛡️ AI Risk Manager")
st.caption(
    "Defense-only AI system for fraud, return/RTO, chargeback and financial-abuse risk."
)

with st.sidebar:
    st.header("⚙️ Transaction Input")

    st.subheader("Transaction")
    amount = st.number_input("Transaction Amount (₹)", 1.0, 1_000_000.0, 2500.0, 100.0)
    amount_deviation = st.slider("Amount Deviation", 0.0, 10.0, 2.0, 0.1)
    payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Food", "Electronics", "Travel", "Fashion", "Gaming", "Grocery", "Services"],
    )

    st.subheader("Velocity")
    transactions_last_10min = st.number_input("Transactions - Last 10 Minutes", 0, 100, 1)
    transactions_last_1h = st.number_input("Transactions - Last 1 Hour", 0, 500, 2)
    transactions_last_24h = st.number_input("Transactions - Last 24 Hours", 0, 2000, 5)

    st.subheader("Device & Location")
    device_age_days = st.number_input("Device Age (days)", 0, 5000, 500)
    device_transaction_count = st.number_input("Device Transaction Count", 0, 5000, 2)
    location_distance_km = st.number_input("Location Distance (km)", 0.0, 10000.0, 10.0, 1.0)
    ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.1, 0.01)

    st.subheader("Security")
    failed_attempts_24h = st.number_input("Failed Attempts - Last 24h", 0, 100, 0)
    previous_fraud_count = st.number_input("Previous Fraud Count", 0, 100, 0)
    chargeback_history = st.number_input("Chargeback History", 0, 100, 0)

    st.subheader("Customer")
    customer_age_days = st.number_input("Customer Age (days)", 1, 10000, 1000)
    customer_avg_amount = st.number_input("Customer Average Amount (₹)", 0.0, 1_000_000.0, 2200.0, 100.0)
    account_velocity_score = st.slider("Account Velocity Score", 0.0, 1.0, 0.2, 0.01)

    st.subheader("Return / RTO")
    order_count = st.number_input("Total Orders", 1, 10000, 10)
    return_count = st.number_input("Previous Returns", 0, 10000, 0)
    rto_count = st.number_input("Previous RTOs", 0, 10000, 0)
    failed_delivery_count = st.number_input("Failed Deliveries", 0, 10000, 0)
    cod_orders = st.number_input("COD Orders", 0, 10000, 2)

    st.subheader("Abuse-Ring Signals")
    linked_accounts = st.number_input("Linked Accounts", 0, 100, 1)
    shared_device_accounts = st.number_input("Accounts Sharing Device", 0, 100, 1)
    shared_ip_accounts = st.number_input("Accounts Sharing IP", 0, 100, 1)

    st.subheader("Time")
    transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
    is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    assess = st.button("🔍 Assess Transaction", type="primary", use_container_width=True)

transaction = {
    "amount": amount, "amount_deviation": amount_deviation,
    "payment_method": payment_method, "merchant_category": merchant_category,
    "transactions_last_10min": transactions_last_10min,
    "transactions_last_1h": transactions_last_1h,
    "transactions_last_24h": transactions_last_24h,
    "device_age_days": device_age_days,
    "device_transaction_count": device_transaction_count,
    "location_distance_km": location_distance_km,
    "ip_risk_score": ip_risk_score,
    "failed_attempts_24h": failed_attempts_24h,
    "previous_fraud_count": previous_fraud_count,
    "customer_age_days": customer_age_days,
    "customer_avg_amount": customer_avg_amount,
    "account_velocity_score": account_velocity_score,
    "chargeback_history": chargeback_history,
    "transaction_hour": transaction_hour, "is_weekend": is_weekend,
    "order_count": order_count, "return_count": return_count,
    "rto_count": rto_count, "failed_delivery_count": failed_delivery_count,
    "cod_orders": cod_orders, "linked_accounts": linked_accounts,
    "shared_device_accounts": shared_device_accounts,
    "shared_ip_accounts": shared_ip_accounts,
}

if assess:
    fraud = calculate_risk(transaction)
    ret = calculate_return_risk(transaction)
    cb = calculate_chargeback_risk(transaction, fraud)
    abuse = calculate_abuse_ring_risk(transaction)

    st.session_state["assessment"] = {
        "fraud": fraud, "return": ret, "chargeback": cb, "abuse": abuse,
        "transaction": transaction,
    }
    save_transaction(transaction, fraud, ret, cb, abuse)
    st.success("Transaction assessed and saved to the local risk history.")

assessment = st.session_state.get("assessment")

if assessment:
    fraud = assessment["fraud"]
    ret = assessment["return"]
    cb = assessment["chargeback"]
    abuse = assessment["abuse"]
    tx = assessment["transaction"]

    st.header("📊 AI Risk Assessment")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{fraud['fraud_probability'] * 100:.2f}%")
    c2.metric("Overall Risk", f"{fraud['risk_score']}/100")
    c3.metric("Risk Level", fraud["risk_level"])
    c4.metric("Decision", fraud["recommended_action"])

    st.subheader("🧩 Loss-Prevention Risk Matrix")
    matrix = pd.DataFrame({
        "Risk Type": ["Fraud", "Return/RTO", "Chargeback", "Abuse Ring"],
        "Score": [fraud["risk_score"], ret["score"], cb["score"], abuse["score"]],
    })
    matrix["Level"] = matrix["Score"].apply(level)
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        fig = px.bar(matrix, x="Risk Type", y="Score", text="Score", range_y=[0, 100],
                     title="Risk by Loss Category")
        fig.add_hline(y=30, line_dash="dash", annotation_text="Medium threshold")
        fig.add_hline(y=70, line_dash="dash", annotation_text="High threshold")
        st.plotly_chart(fig, use_container_width=True)
    with mc2:
        st.dataframe(matrix, hide_index=True, use_container_width=True)

    st.subheader("🔎 Explainable Risk Signals")
    for reason in fraud["risk_reasons"]:
        st.warning(f"⚠️ {reason}")

    with st.expander("↩️ Return / RTO Risk"):
        st.metric("Return/RTO Score", f"{ret['score']}/100")
        st.write(f"Return rate: {ret['return_rate']:.1%}")
        st.write(f"RTO rate: {ret['rto_rate']:.1%}")
        for r in ret["reasons"]:
            st.write(f"• {r}")

    with st.expander("💳 Chargeback Evidence Responder"):
        st.metric("Chargeback Risk", f"{cb['score']}/100")
        evidence = build_chargeback_evidence(tx, fraud, cb)
        st.write(evidence["summary"])
        for item in evidence["evidence_items"]:
            st.write(f"☑️ {item}")
        st.info(evidence["recommended_next_step"])
        evidence_text = "\n".join(
            ["CHARGEBACK EVIDENCE CHECKLIST", *[f"- {x}" for x in evidence["evidence_items"]],
             "", f"AI Risk Score: {fraud['risk_score']}", f"Chargeback Risk: {cb['score']}",
             f"Decision: {fraud['recommended_action']}"]
        )
        st.download_button(
            "📥 Download Evidence Checklist",
            evidence_text,
            "chargeback_evidence_checklist.txt",
            "text/plain",
            use_container_width=True,
        )

    with st.expander("🕸️ Abuse-Ring Sentinel"):
        st.metric("Abuse-Ring Risk", f"{abuse['score']}/100")
        for r in abuse["reasons"]:
            st.write(f"• {r}")
        st.caption("Defense-only heuristic. It flags patterns for review; it does not perform offensive activity.")

    st.subheader("🎯 Risk Score Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud["risk_score"],
        title={"text": "Overall Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "#22c55e"},
                {"range": [30, 70], "color": "#f59e0b"},
                {"range": [70, 100], "color": "#ef4444"},
            ],
        },
    ))
    gauge.update_layout(height=350)
    st.plotly_chart(gauge, use_container_width=True)

history = load_history()
if not history.empty:
    for col in ["risk_score", "fraud_probability", "return_risk_score",
                "chargeback_risk_score", "abuse_ring_risk_score"]:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)

st.divider()
st.header("📈 Risk Analytics & Evaluation")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fraud Spike", "Risk Distribution", "Merchant Analysis",
    "Model Evaluation", "Transaction History"
])

with tab1:
    spike = detect_fraud_spike(history)
    a, b, c = st.columns(3)
    a.metric("Fraud-Spike Status", spike["status"])
    b.metric("Recent High-Risk Rate", f"{spike['recent_rate']:.1%}")
    c.metric("Baseline High-Risk Rate", f"{spike['baseline_rate']:.1%}")
    if spike["status"] == "SPIKE DETECTED":
        st.error("🚨 FRAUD SPIKE DETECTED — recent high-risk activity is materially above baseline.")
    elif spike["status"] == "ELEVATED":
        st.warning("⚠️ Elevated high-risk activity.")
    else:
        st.success("🟢 No material fraud-risk spike detected.")
    st.write(spike["message"])

    if len(history) >= 2:
        trend = history.tail(100).reset_index()
        trend["Transaction"] = trend.index + 1
        chart = px.line(trend, x="Transaction", y="risk_score", title="Recent Risk Score Trend")
        st.plotly_chart(chart, use_container_width=True)

with tab2:
    if history.empty:
        st.info("Assess transactions to populate risk distribution.")
    else:
        counts = history["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0).reset_index()
        counts.columns = ["Risk Level", "Transactions"]
        fig = px.pie(counts, names="Risk Level", values="Transactions", hole=0.4,
                     title="Transaction Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if history.empty:
        st.info("Assess transactions to populate merchant analysis.")
    else:
        merchant = history.groupby("merchant_category").agg(
            Transactions=("risk_score", "count"),
            Average_Risk=("risk_score", "mean"),
            Average_Fraud_Probability=("fraud_probability", "mean"),
        ).reset_index()
        merchant["Average_Risk"] = merchant["Average_Risk"].round(2)
        merchant["Average_Fraud_Probability"] = (merchant["Average_Fraud_Probability"] * 100).round(2)
        st.dataframe(merchant, hide_index=True, use_container_width=True)
        fig = px.bar(merchant, x="merchant_category", y="Average_Risk",
                     text="Average_Risk", title="Average Risk by Merchant Category")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🧪 Held-Out Model Evaluation")
    st.caption("Values below are the Random Forest evaluation recorded in the project's 80/20 held-out test notebook.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "99.84%")
    m2.metric("Precision", "97.97%")
    m3.metric("Recall", "98.53%")
    m4.metric("F1 Score", "98.25%")

    st.write("**Test set:** 20,000 unseen transactions (19,117 legitimate; 883 fraud).")
    eval_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        "Score": ["99.84%", "97.97%", "98.53%", "98.25%", "99.99%", "99.86%"],
    })
    st.dataframe(eval_df, hide_index=True, use_container_width=True)

    st.subheader("💰 False-Positive Cost Control")
    fp_cost = st.number_input("Estimated cost per false positive (₹)", 0.0, 1_000_000.0, 500.0, 100.0)
    # The notebook's reported precision/recall on 883 fraud cases implies approximately
    # 13 false negatives and 18 false positives after rounding.
    estimated_fp = 18
    estimated_fn = 13
    st.metric("Estimated False-Positive Cost", f"₹{estimated_fp * fp_cost:,.0f}")
    st.write(f"Evaluation reference: approximately {estimated_fp} false positives and {estimated_fn} false negatives.")
    st.caption("Use the exact confusion matrix from a fresh model run if you need audit-grade counts; displayed counts are derived from rounded notebook metrics.")

with tab5:
    if history.empty:
        st.info("No transaction history available yet.")
    else:
        display = history.sort_values("timestamp", ascending=False).copy()
        cols = [
            "timestamp", "amount", "merchant_category", "fraud_probability",
            "risk_score", "risk_level", "recommended_action",
            "return_risk_score", "chargeback_risk_score", "abuse_ring_risk_score",
        ]
        display = display[cols]
        display["fraud_probability"] = (display["fraud_probability"] * 100).round(2).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Complete Risk Report",
            csv,
            "AI_Risk_Manager_Complete_Risk_Report.csv",
            "text/csv",
            use_container_width=True,
        )

st.divider()
st.subheader("🛡️ Defense-Only Safety Boundary")
st.info(
    "This application is designed for defensive loss prevention: detect suspicious "
    "transactions, score risk, support manual verification, and recommend approve/review/block decisions. "
    "It does not perform offensive security activity."
)
st.caption("AI Risk Manager • Razorpay AI Buildathon 2026 • Track 02")
