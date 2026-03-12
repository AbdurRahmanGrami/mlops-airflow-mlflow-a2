# 🚢 Titanic MLOps Pipeline
### End-to-End Machine Learning Pipeline with Apache Airflow + MLflow

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8.1-017CEE)
![MLflow](https://img.shields.io/badge/MLflow-2.17.2-0194E2)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## 📖 Overview

This project implements a fully automated end-to-end MLOps pipeline that predicts passenger survival on the Titanic dataset. Apache Airflow orchestrates the workflow through a DAG with parallel tasks and branching logic, while MLflow handles experiment tracking, metric logging, and model registration.

**Dataset:** [Titanic Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data)

---

## 🏗️ Architecture

```
Apache Airflow (localhost:8080)          MLflow Server (localhost:5000)
────────────────────────────────         ──────────────────────────────
DAG: titanic_mlops_pipeline              Experiment: titanic_survival_prediction
│                                        │
├── ingest_data                          ├── Run 1: RandomForest (n=100, depth=5)
├── validate_data (retries=2)            ├── Run 2: RandomForest (n=200, depth=10)
├── handle_missing_values ──┐            ├── Run 3: LogisticRegression (C=1.0)
├── feature_engineering  ──┤ parallel   │
├── encode_data             │            └── Model Registry: titanic_survival_model
├── train_model  ───────────┼──────────► logs params, metrics, artifacts
├── evaluate_model          │
├── branch_decision         │
│   ├── register_model ─────┼──────────► registers to Model Registry
│   └── reject_model        │
└── end                     │
```

---

## 📁 Project Structure

```
titanic-mlops/
│
├── dags/
│   └── mlops_airflow_mlflow_pipeline.py   # Main DAG file
│
├── data/
│   └── Titanic-Dataset.csv                # Raw dataset
│
├── logs/                                  # Airflow task logs (auto-generated)
├── plugins/                               # Airflow plugins (empty)
│
├── docker-compose.yaml                    # Airflow multi-container setup
├── .env                                   # AIRFLOW_UID (auto-generated)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## ✅ Prerequisites

Make sure you have the following installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.8+](https://www.python.org/downloads/)
- [MLflow](https://mlflow.org/) (`pip install mlflow`)
- Titanic dataset downloaded from Kaggle

---

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AbdurRahmanGrami/titanic-mlops.git
cd titanic-mlops
```

### 2. Download the Titanic Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data) and place it in the data folder:

```bash
mkdir -p data
mv /path/to/Titanic-Dataset.csv data/
```

### 3. Set Airflow UID (Linux only)

```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

### 4. Find Your Docker Bridge IP

```bash
ip addr show docker0 | grep "inet "
# Example output: inet 172.17.0.1/16
```

Update this IP in two places:

**In `docker-compose.yaml`:**
```yaml
MLFLOW_TRACKING_URI: http://172.17.0.1:5000
```

**In `dags/mlops_airflow_mlflow_pipeline.py`:**
```python
MLFLOW_TRACKING_URI = "http://172.17.0.1:5000"
```

### 5. Start MLflow Server

Open a dedicated terminal and keep it running:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Verify at: **http://localhost:5000**

### 6. Initialize and Start Airflow

```bash
docker-compose up airflow-init
docker-compose up -d
```

Check all services are healthy:

```bash
docker-compose ps
```

### 7. Install Python Packages Inside Containers

```bash
docker-compose exec airflow-worker /home/airflow/.local/bin/pip install mlflow scikit-learn
docker-compose exec airflow-scheduler /home/airflow/.local/bin/pip install mlflow scikit-learn
docker-compose exec airflow-webserver /home/airflow/.local/bin/pip install mlflow scikit-learn
```

### 8. Allow Docker to Reach MLflow (Linux firewall)

```bash
sudo ufw allow from 172.16.0.0/12 to any port 5000
```

---

## ▶️ Running the Pipeline

1. Open Airflow UI at **http://localhost:8080**
   - Username: `airflow`
   - Password: `airflow`

2. Find `titanic_mlops_pipeline` and toggle it **ON**

3. Click the **▶ Trigger DAG** button

4. Watch tasks execute in the **Graph** view

---

## 🔬 Running Experiment Comparison (Task 10)

Run the DAG **3 times** with different hyperparameters by editing `train_model_fn` in the DAG file:

**Run 1 — Random Forest (baseline):**
```python
model_type = "RandomForest"
hyperparams = { "n_estimators": 100, "max_depth": 5, "random_state": 42 }
```

**Run 2 — Random Forest (tuned):**
```python
model_type = "RandomForest"
hyperparams = { "n_estimators": 200, "max_depth": 10, "random_state": 42 }
```

**Run 3 — Logistic Regression:**
```python
model_type = "LogisticRegression"
hyperparams = { "max_iter": 200, "C": 1.0, "random_state": 42 }
```

Compare all runs at **http://localhost:5000** → select all runs → click **Compare**.

---

## 📊 Experiment Results

| Run | Model | n_estimators | max_depth | Accuracy | F1-Score | Status |
|-----|-------|-------------|-----------|----------|----------|--------|
| Run 1 | Random Forest | 100 | 5 | — | — | Registered |
| Run 2 | Random Forest | 200 | 10 | 0.827 | 0.780 | Registered |
| Run 3 | Logistic Regression | — | C=1.0 | 0.799 | 0.746 | Rejected |

**Best Model:** Run 2 — Random Forest with n_estimators=200, max_depth=10

---

## 🔄 DAG Task Summary

| Task | Description |
|------|-------------|
| `ingest_data` | Loads CSV, prints shape, logs missing values, pushes path via XCom |
| `validate_data` | Validates missing % for Age/Embarked, raises exception if > 30%, retries=2 |
| `handle_missing_values` | Fills Age (median), Embarked (mode), Cabin (Unknown) — runs in parallel |
| `feature_engineering` | Creates FamilySize and IsAlone features — runs in parallel |
| `encode_data` | Encodes Sex/Embarked, drops irrelevant columns, merges parallel outputs |
| `train_model` | Trains model, logs params and artifact to MLflow |
| `evaluate_model` | Computes Accuracy, Precision, Recall, F1 — logs to MLflow |
| `branch_decision` | Routes to register if accuracy ≥ 0.80, else reject |
| `register_model` | Registers model in MLflow Model Registry with tags |
| `reject_model` | Logs rejection reason and status tag to MLflow |

---

## 🛑 Stopping the Pipeline

```bash
# Stop Airflow containers
docker-compose down

# Stop MLflow (Ctrl+C in its terminal, then)
pkill -f "mlflow server"
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| DAG import error: No module named mlflow | Run pip install inside containers (Step 7 above) |
| Docker can't reach MLflow | Run `sudo ufw allow from 172.16.0.0/12 to any port 5000` |
| Permission denied on pip | Use `/home/airflow/.local/bin/pip` instead of `pip` |
| DAG not appearing in UI | Check for syntax errors: `python dags/mlops_airflow_mlflow_pipeline.py` |
| XCom returning None | Ensure upstream task completed successfully before downstream runs |

---

## 👤 Author

**Abdur Rahman**
GitHub: [@AbdurRahmanGrami](https://github.com/AbdurRahmanGrami)

---

## 📄 License

This project is submitted as academic coursework for the MLOps course, BS Data Science.
