import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_model.pkl"
model = joblib.load(MODEL_PATH)

MODEL_FIELDS = [
    "amount", "payment_method", "merchant_category", "transaction_hour",
    "is_weekend", "customer_age_days", "customer_avg_amount",
    "amount_deviation", "transactions_last_10min", "transactions_last_1h",
    "transactions_last_24h", "device_age_days", "device_transaction_count",
    "location_distance_km", "ip_risk_score", "failed_attempts_24h",
    "previous_fraud_count", "chargeback_history", "account_velocity_score",
]


def calculate_risk(transaction):
    """ML fraud probability + explainable rule-based signals."""
    df = pd.DataFrame([{k: transaction[k] for k in MODEL_FIELDS}])
    ml_probability = float(model.predict_proba(df)[0][1])

    checks = [
        (transaction["amount"] > 5000, "High transaction amount", 10),
        (transaction["amount_deviation"] > 4,
         "Transaction amount is significantly unusual", 10),
        (transaction["transactions_last_10min"] >= 4,
         "High transaction velocity in the last 10 minutes", 15),
        (transaction["transactions_last_1h"] >= 8,
         "High transaction velocity in the last hour", 10),
        (transaction["transactions_last_24h"] >= 20,
         "High transaction activity in the last 24 hours", 10),
        (transaction["device_age_days"] < 30, "Very new device", 10),
        (transaction["device_transaction_count"] > 20,
         "High number of transactions from this device", 10),
        (transaction["location_distance_km"] > 300,
         "Unusually large location distance", 10),
        (transaction["ip_risk_score"] > 0.6, "High-risk IP address", 15),
        (transaction["failed_attempts_24h"] >= 3,
         "Multiple failed attempts", 10),
        (transaction["previous_fraud_count"] > 0,
         "Previous fraud history detected", 20),
        (transaction["chargeback_history"] > 0,
         "Previous chargeback history", 15),
        (transaction["account_velocity_score"] > 0.7,
         "Unusually high account velocity", 15),
    ]

    reasons = [reason for active, reason, _ in checks if active]
    rule_score = min(sum(weight for active, _, weight in checks if active), 100)

    ml_score = ml_probability * 100
    final_score = round(min(0.70 * ml_score + 0.30 * rule_score, 100), 2)

    if final_score < 30:
        level, action = "LOW", "APPROVE"
    elif final_score < 70:
        level, action = "MEDIUM", "MANUAL REVIEW"
    else:
        level, action = "HIGH", "BLOCK"

    return {
        "ml_probability": round(ml_probability, 4),
        "rule_signal_score": rule_score,
        "fraud_probability": round(final_score / 100, 4),
        "risk_score": final_score,
        "risk_level": level,
        "recommended_action": action,
        "risk_reasons": reasons or ["No major risk indicators detected"],
    }


def calculate_return_risk(transaction):
    """Dedicated Return/RTO loss-risk scorer."""
    orders = max(int(transaction.get("order_count", 1)), 1)
    returns = int(transaction.get("return_count", 0))
    rto = int(transaction.get("rto_count", 0))
    failed_delivery = int(transaction.get("failed_delivery_count", 0))
    cod = int(transaction.get("cod_orders", 0))

    return_rate = min(returns / orders, 1.0)
    rto_rate = min(rto / orders, 1.0)
    delivery_failure_rate = min(failed_delivery / orders, 1.0)
    cod_rate = min(cod / orders, 1.0)

    score = (
        return_rate * 35
        + rto_rate * 30
        + delivery_failure_rate * 20
        + cod_rate * 10
    )
    if transaction.get("amount", 0) > 5000:
        score += 5

    score = round(min(score, 100), 2)
    level = "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"

    reasons = []
    if return_rate >= 0.30:
        reasons.append(f"High return rate ({return_rate:.0%})")
    if rto_rate >= 0.20:
        reasons.append(f"High RTO rate ({rto_rate:.0%})")
    if delivery_failure_rate >= 0.15:
        reasons.append(f"Repeated delivery failures ({failed_delivery})")
    if cod_rate >= 0.70:
        reasons.append("Heavy cash-on-delivery usage")
    if not reasons:
        reasons.append("No major return/RTO indicators detected")

    return {
        "score": score,
        "level": level,
        "return_rate": return_rate,
        "rto_rate": rto_rate,
        "delivery_failure_rate": delivery_failure_rate,
        "reasons": reasons,
    }


def calculate_chargeback_risk(transaction, fraud_result=None):
    """Chargeback risk scorer."""
    score = 0
    reasons = []

    if int(transaction.get("chargeback_history", 0)) > 0:
        score += 40
        reasons.append("Previous chargeback history")
    if int(transaction.get("previous_fraud_count", 0)) > 0:
        score += 25
        reasons.append("Previous fraud history")
    if int(transaction.get("failed_attempts_24h", 0)) >= 3:
        score += 10
        reasons.append("Multiple failed attempts")
    if float(transaction.get("ip_risk_score", 0)) > 0.6:
        score += 15
        reasons.append("High-risk IP")
    if float(transaction.get("account_velocity_score", 0)) > 0.7:
        score += 10
        reasons.append("High account velocity")
    if fraud_result and fraud_result.get("risk_score", 0) >= 70:
        score += 10
        reasons.append("High overall transaction risk")

    score = round(min(score, 100), 2)
    level = "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"

    return {
        "score": score,
        "level": level,
        "reasons": reasons or ["No major chargeback indicators detected"],
    }


def calculate_abuse_ring_risk(transaction):
    """Defense-only linked-account/device/IP abuse sentinel."""
    score = 0
    reasons = []

    linked = int(transaction.get("linked_accounts", 0))
    shared_device = int(transaction.get("shared_device_accounts", 0))
    shared_ip = int(transaction.get("shared_ip_accounts", 0))

    if linked >= 3:
        score += 30
        reasons.append(f"{linked} linked accounts detected")
    if shared_device >= 3:
        score += 30
        reasons.append("Multiple accounts share the same device")
    if shared_ip >= 4:
        score += 25
        reasons.append("Multiple accounts share the same IP")
    if int(transaction.get("device_transaction_count", 0)) > 20:
        score += 10
        reasons.append("High transaction activity from device")
    if float(transaction.get("account_velocity_score", 0)) > 0.7:
        score += 15
        reasons.append("High account velocity")

    score = round(min(score, 100), 2)
    level = "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"

    return {
        "score": score,
        "level": level,
        "reasons": reasons or ["No major abuse-ring indicators detected"],
    }


def detect_fraud_spike(history_df, window=20, baseline_window=50):
    """Detect a recent increase in HIGH-risk transactions."""
    if history_df is None or len(history_df) < 10:
        return {
            "status": "INSUFFICIENT DATA",
            "score": 0,
            "recent_rate": 0,
            "baseline_rate": 0,
            "message": "Assess at least 10 transactions to detect a trend.",
        }

    df = history_df.copy()
    recent = df.tail(window)
    baseline = df.iloc[:-window].tail(baseline_window)

    if baseline.empty:
        return {
            "status": "INSUFFICIENT DATA",
            "score": 0,
            "recent_rate": 0,
            "baseline_rate": 0,
            "message": "More historical transactions are required for a baseline.",
        }

    recent_rate = float((recent["risk_level"] == "HIGH").mean())
    baseline_rate = float((baseline["risk_level"] == "HIGH").mean())
    increase = recent_rate - baseline_rate
    ratio = recent_rate / max(baseline_rate, 0.01)
    score = round(min(max(increase * 200 + max(ratio - 1, 0) * 20, 0), 100), 2)

    if recent_rate >= 0.20 and increase >= 0.10:
        status = "SPIKE DETECTED"
    elif increase >= 0.05:
        status = "ELEVATED"
    else:
        status = "NORMAL"

    return {
        "status": status,
        "score": score,
        "recent_rate": recent_rate,
        "baseline_rate": baseline_rate,
        "message": (
            "Recent high-risk activity is materially above the baseline."
            if status == "SPIKE DETECTED"
            else "No material fraud-risk spike detected."
        ),
    }


def build_chargeback_evidence(transaction, fraud_result, chargeback_result):
    """Generate a defensive, review-ready chargeback evidence checklist."""
    return {
        "summary": "Defensive chargeback evidence checklist generated.",
        "risk_score": fraud_result.get("risk_score", 0),
        "chargeback_score": chargeback_result.get("score", 0),
        "risk_reasons": fraud_result.get("risk_reasons", []),
        "evidence_items": [
            "Transaction amount, timestamp and payment method",
            "Merchant category and customer/account history",
            "Device age and device transaction activity",
            "Location distance and IP risk evidence",
            "Authentication and failed-attempt logs",
            "Previous fraud and chargeback history",
            "10-minute, 1-hour and 24-hour velocity signals",
            "AI risk score, fraud probability and decision",
            "Delivery/service proof and customer communication, where applicable",
        ],
        "recommended_next_step": (
            "Review source logs and supporting proof before submitting "
            "a chargeback response."
        ),
    }
