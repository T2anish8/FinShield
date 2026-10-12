# 🛡️ FinShield AI — Financial Risk & Fraud Intelligence Platform

FinShield AI is an educational financial intelligence platform designed to explore **fraud detection, Anti-Money Laundering (AML), anomaly detection, and graph-based transaction risk analysis**. It combines machine learning, transaction feature engineering, rule-based monitoring, and network intelligence in a unified risk analysis workflow.

The project uses synthetic financial transaction data to demonstrate how suspicious patterns can be identified and investigated through a web-based dashboard.

## ✨ Key Features

### 🔍 Fraud Detection
- Train and evaluate fraud classification models using **Random Forest and XGBoost**.
- Engineer transaction-level features to identify potentially suspicious activity.
- Evaluate models using precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Analyze classification thresholds to explore the trade-off between false positives and fraud detection.

### 🏦 Anti-Money Laundering (AML)
- Apply rule-based checks to identify potentially suspicious transaction patterns.
- Explore scenarios such as structuring, layering, mule accounts, circular transfers, dormant account activation, and rapid fund movement.
- Generate AML alerts and assign risk levels for further investigation.

### 🚨 Anomaly Detection
- Use **Isolation Forest** to identify unusual transaction behaviour.
- Explore deviations from normal transaction patterns.
- Combine anomaly signals with other risk indicators for broader analysis.

### 🕸️ Graph-Based Network Intelligence
- Analyze relationships between accounts and transactions.
- Explore connections involving shared devices, shared IP addresses, and suspicious transfers.
- Identify potentially high-risk accounts and connected suspicious activity.
- Support investigation through network-based risk insights.

### 📊 Unified Risk Intelligence
- Combine multiple risk signals into a consolidated transaction or account risk view.
- Generate processed risk datasets and summary outputs.
- Support investigation prioritization through risk categories and alerts.

### 💻 Interactive Dashboard
- Frontend built with **React, TypeScript, Vite, and Tailwind CSS**.
- Backend APIs built with **FastAPI**.
- Provides a web interface for exploring financial risk intelligence and network analysis.

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python, TypeScript, JavaScript |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, Tailwind CSS |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Graph Analysis | NetworkX |
| Model Persistence | joblib |
| Synthetic Data | Faker |
| Version Control | Git, GitHub |

## 🏗️ Project Structure

```text
FinShield/
├── backend/
│   ├── main.py
│   └── transaction_simulator.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── NetworkIntelligence.tsx
│   ├── package.json
│   └── vite.config.ts
├── src/
│   ├── aml/
│   │   └── aml_rule_engine.py
│   ├── anomaly/
│   │   └── anomaly_detector.py
│   ├── data_generation/
│   │   ├── generator.py
│   │   ├── inject_aml.py
│   │   └── inject_fraud.py
│   ├── features/
│   │   └── transaction_features.py
│   ├── fraud/
│   │   ├── evaluate_thresholds.py
│   │   └── train_fraud_model.py
│   ├── graph/
│   │   └── graph_intelligence.py
│   ├── risk/
│   │   └── unified_risk_engine.py
│   └── config.py
├── data/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.10 or later
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/T2anish8/FinShield.git
cd FinShield
```

### 2. Set up the Python environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the commands below without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Start the backend

From the project root, run:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Once the server starts, open the interactive API documentation:

**http://127.0.0.1:8000/docs**

### 4. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm ci
npm.cmd run dev
```

Open the local URL printed by Vite in your terminal, typically:

**http://localhost:5173**

Keep both the backend and frontend running while using the application.

## 🔬 Machine Learning Workflow

The project explores the following workflow:

1. **Synthetic data generation** — create representative transaction data.
2. **Scenario injection** — introduce simulated fraud and AML patterns.
3. **Feature engineering** — derive transaction-level features.
4. **Model training and evaluation** — train and assess fraud detection models.
5. **AML rule evaluation** — identify transactions that meet configured suspicious-activity rules.
6. **Anomaly detection** — explore unusual patterns using Isolation Forest.
7. **Graph intelligence** — examine relationships between accounts and suspicious transactions.
8. **Unified risk analysis** — consolidate available risk signals for investigation.

Individual pipeline scripts may require earlier stages to have generated their expected input files. Run them in the appropriate order and verify the configured data paths before execution.

## 🔐 Data, Privacy & Responsible Use

- This project is designed for educational and portfolio demonstration.
- Synthetic data is used to simulate financial transactions and suspicious scenarios.
- Synthetic patterns do not necessarily represent real-world financial crime.
- Model predictions and rule-based alerts are indicators for investigation, not proof of wrongdoing.
- The platform has not been established as a production-ready AML compliance or fraud prevention system.

Do not upload real customer financial data, credentials, API keys, or other sensitive information to a public repository.

## 🚀 Future Improvements

- Add explainable AI to show why transactions receive elevated risk scores.
- Improve model evaluation using robust validation and class-imbalance techniques.
- Add account-level investigation timelines and richer graph exploration.
- Introduce automated model monitoring and data-drift detection.
- Add unit tests, API integration tests, and continuous integration.
- Improve deployment, authentication, logging, and security controls.

## 👨‍💻 Author

**Tanish Mittal**

B.E. Data Science | AI/ML | Financial Intelligence

GitHub: [@T2anish8](https://github.com/T2anish8)

## 📄 License

Choose and add an appropriate open-source license before permitting reuse or redistribution. Until a license is added, the repository should not be assumed to grant permission to reuse the code.
