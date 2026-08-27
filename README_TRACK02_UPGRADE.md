# AI Risk Manager — Track 02 Upgrade

This upgrade aligns the Streamlit demo with the Razorpay AI Buildathon Track 02 requirements.

## Added
- Fraud-spike detector
- Dedicated Return/RTO risk scorer
- Chargeback risk scorer
- Defensive chargeback evidence responder/checklist
- Abuse-ring sentinel using linked account/device/IP signals
- Detect → Verify → Respond decision flow
- Approve / Manual Review / Block recommendations
- Held-out model evaluation: precision, recall, accuracy and F1
- False-positive cost control
- Risk matrix across Fraud, Return/RTO, Chargeback and Abuse Ring
- Risk analytics and downloadable evidence/report
- Explicit defense-only boundary

## Important
The return/RTO and abuse-ring modules are transparent rule-based risk heuristics. They are not presented as independently trained ML models.

## Existing model
The application continues to load `models/risk_model.pkl`, so keep that file in the repository.

## Streamlit
Main file: `app.py`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
On Streamlit Community Cloud:
- Repository: `Elactic2702/AI-Risk-Manager`
- Branch: `main`
- Main file: `AI-Risk-Manager/app.py`
- Python dependencies: `AI-Risk-Manager/requirements.txt`
