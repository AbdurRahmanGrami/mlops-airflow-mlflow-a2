from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient

# ──────────────────────────────────────────────
# Default arguments for every task in the DAG
# ──────────────────────────────────────────────
default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,                          # Task 3 needs retry evidence
    'retry_delay': timedelta(seconds=30),
}

# ──────────────────────────────────────────────
# DAG Definition
# ──────────────────────────────────────────────

import pandas as pd
import os

def ingest_data_fn(**context):
    # ── Path to the mounted CSV file ──────────────────────
    dataset_path = '/opt/airflow/data/Titanic-Dataset.csv'

    # ── Check file exists ─────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # ── Load dataset ──────────────────────────────────────
    df = pd.read_csv(dataset_path)

    # ── Print dataset shape ───────────────────────────────
    print(f"Dataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # ── Log missing values per column ─────────────────────
    missing = df.isnull().sum()
    print("\n── Missing Values Per Column ──")
    for col, count in missing.items():
        print(f"  {col}: {count} missing")

    # ── Push dataset path via XCom ────────────────────────
    context['ti'].xcom_push(key='dataset_path', value=dataset_path)
    print(f"\nXCom pushed: dataset_path = {dataset_path}")

def validate_data_fn(**context):
    # ── Pull dataset path from XCom ───────────────────────
    ti = context['ti']
    dataset_path = ti.xcom_pull(task_ids='ingest_data', key='dataset_path')

    if not dataset_path:
        raise ValueError("No dataset path received from ingest_data via XCom")

    # ── INTENTIONAL FAILURE FOR RETRY DEMONSTRATION ───────
    # Get how many times this task has been tried
    try_number = context['ti'].try_number
    print(f"Attempt number: {try_number}")

    if try_number < 3:
        raise Exception(
            f"Intentional failure on attempt {try_number} — simulating retry behavior"
        )

    # ── (Reaches here only on 3rd attempt) ────────────────
    print("✓ Retry demonstration complete — proceeding with real validation")

    # ── Load dataset ──────────────────────────────────────
    df = pd.read_csv(dataset_path)

    total_rows = len(df)
    age_missing_pct = (df['Age'].isnull().sum() / total_rows) * 100
    embarked_missing_pct = (df['Embarked'].isnull().sum() / total_rows) * 100

    print(f"\n── Missing Value Percentages ──")
    print(f"  Age:      {age_missing_pct:.2f}%")
    print(f"  Embarked: {embarked_missing_pct:.2f}%")

    # ── Raise exception if missing > 30% ──────────────────
    if age_missing_pct > 30:
        raise ValueError(
            f"VALIDATION FAILED: Age missing {age_missing_pct:.2f}% — exceeds 30% threshold"
        )

    if embarked_missing_pct > 30:
        raise ValueError(
            f"VALIDATION FAILED: Embarked missing {embarked_missing_pct:.2f}% — exceeds 30% threshold"
        )

    print("\n✓ Validation passed — missing values within acceptable range")

    ti.xcom_push(key='validation_status', value='passed')  


def handle_missing_fn(**context):
    ti = context['ti']
    dataset_path = ti.xcom_pull(task_ids='ingest_data', key='dataset_path')

    # ── Load dataset ──────────────────────────────────────
    df = pd.read_csv(dataset_path)

    print("── Before Handling Missing Values ──")
    print(f"  Age missing:      {df['Age'].isnull().sum()}")
    print(f"  Embarked missing: {df['Embarked'].isnull().sum()}")

    # ── Fill Age with median ──────────────────────────────
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"\n✓ Age filled with median: {age_median:.1f}")

    # ── Fill Embarked with mode ───────────────────────────
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"✓ Embarked filled with mode: {embarked_mode}")

    # ── Fill Cabin with 'Unknown' ─────────────────────────
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    print(f"✓ Cabin filled with: Unknown")

    print("\n── After Handling Missing Values ──")
    print(f"  Age missing:      {df['Age'].isnull().sum()}")
    print(f"  Embarked missing: {df['Embarked'].isnull().sum()}")
    print(f"  Cabin missing:    {df['Cabin'].isnull().sum()}")

    # ── Save cleaned data and push path via XCom ──────────
    cleaned_path = '/opt/airflow/data/titanic_cleaned.csv'
    df.to_csv(cleaned_path, index=False)
    print(f"\n✓ Cleaned dataset saved to: {cleaned_path}")

    ti.xcom_push(key='cleaned_path', value=cleaned_path)

def feature_engineering_fn(**context):
    ti = context['ti']
    dataset_path = ti.xcom_pull(task_ids='ingest_data', key='dataset_path')

    # ── Load dataset ──────────────────────────────────────
    df = pd.read_csv(dataset_path)

    # ── FamilySize = SibSp + Parch + 1 (self) ────────────
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print("✓ FamilySize created")
    print(f"  Sample values: {df['FamilySize'].value_counts().head().to_dict()}")

    # ── IsAlone = 1 if travelling alone, else 0 ───────────
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print(f"\n✓ IsAlone created")
    print(f"  Alone:     {df['IsAlone'].sum()} passengers")
    print(f"  Not alone: {(df['IsAlone'] == 0).sum()} passengers")

    # ── Save engineered data and push path via XCom ───────
    engineered_path = '/opt/airflow/data/titanic_engineered.csv'
    df.to_csv(engineered_path, index=False)
    print(f"\n✓ Engineered dataset saved to: {engineered_path}")

    ti.xcom_push(key='engineered_path', value=engineered_path)    


def encode_data_fn(**context):
    ti = context['ti']

    # ── Pull both paths from parallel tasks ───────────────
    cleaned_path = ti.xcom_pull(task_ids='handle_missing_values', key='cleaned_path')
    engineered_path = ti.xcom_pull(task_ids='feature_engineering', key='engineered_path')

    # ── Load both outputs and merge on PassengerId ────────
    df_cleaned = pd.read_csv(cleaned_path)
    df_engineered = pd.read_csv(engineered_path)

    # Merge to get missing value fixes + new features together
    df = df_cleaned.copy()
    df['FamilySize'] = df_engineered['FamilySize']
    df['IsAlone'] = df_engineered['IsAlone']

    print("── Dataset before encoding ──")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── Encode Sex: male=0, female=1 ──────────────────────
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    print("\n✓ Sex encoded: male=0, female=1")

    # ── Encode Embarked: S=0, C=1, Q=2 ───────────────────
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    print("✓ Embarked encoded: S=0, C=1, Q=2")

    # ── Drop irrelevant columns ───────────────────────────
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=cols_to_drop)
    print(f"\n✓ Dropped columns: {cols_to_drop}")

    print("\n── Dataset after encoding ──")
    print(f"  Shape: {df.shape}")
    print(f"  Remaining columns: {list(df.columns)}")
    print(f"\n  Sample row:\n{df.head(1).to_string()}")

    # ── Save encoded data and push path via XCom ──────────
    encoded_path = '/opt/airflow/data/titanic_encoded.csv'
    df.to_csv(encoded_path, index=False)
    print(f"\n✓ Encoded dataset saved to: {encoded_path}")

    ti.xcom_push(key='encoded_path', value=encoded_path)

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = "http://172.17.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "titanic_survival_prediction"

def train_model_fn(**context):
    ti = context['ti']

    # ── Pull encoded dataset path from XCom ───────────────
    encoded_path = ti.xcom_pull(task_ids='encode_data', key='encoded_path')
    if not encoded_path:
        raise ValueError("No encoded path received from encode_data via XCom")

    # ── Load encoded dataset ──────────────────────────────
    df = pd.read_csv(encoded_path)
    print(f"✓ Loaded encoded dataset: {df.shape}")

    # ── Split features and target ─────────────────────────
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ── Hyperparameters (change these for Task 10 runs) ───
    model_type = "LogisticRegression"
    hyperparams = {
        "max_iter": 200,
        "C": 1.0,
        "random_state": 42,
    }

    # ── Set MLflow tracking URI and experiment ─────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Start MLflow run ──────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n✓ MLflow run started: {run_id}")

        # ── Log model type ─────────────────────────────────
        mlflow.log_param("model_type", model_type)
        print(f"✓ Logged model_type: {model_type}")

        # ── Log hyperparameters ────────────────────────────
        mlflow.log_params(hyperparams)
        print(f"✓ Logged hyperparameters: {hyperparams}")

        # ── Log dataset size ───────────────────────────────
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        print(f"✓ Logged dataset sizes")

        # ── Train model ────────────────────────────────────
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=hyperparams["n_estimators"],
                max_depth=hyperparams["max_depth"],
                random_state=hyperparams["random_state"],
            )
        else:
            model = LogisticRegression(
                max_iter=hyperparams.get("max_iter", 200),
                C=hyperparams.get("C", 1.0),
                random_state=hyperparams["random_state"],
            )

        model.fit(X_train, y_train)
        print(f"\n✓ Model trained: {model_type}")

        # ── Log model artifact ────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"✓ Model artifact logged to MLflow")

        # ── Push run_id and test data path via XCom ───────
        # Save test data for evaluation task
        test_data_path = '/opt/airflow/data/test_data.csv'
        X_test_save = X_test.copy()
        X_test_save['Survived'] = y_test.values
        X_test_save.to_csv(test_data_path, index=False)

        ti.xcom_push(key='run_id', value=run_id)
        ti.xcom_push(key='test_data_path', value=test_data_path)
        ti.xcom_push(key='model_type', value=model_type)
        print(f"\n✓ XCom pushed: run_id, test_data_path, model_type")



def evaluate_model_fn(**context):
    ti = context['ti']

    # ── Pull values from XCom ─────────────────────────────
    run_id = ti.xcom_pull(task_ids='train_model', key='run_id')
    test_data_path = ti.xcom_pull(task_ids='train_model', key='test_data_path')
    model_type = ti.xcom_pull(task_ids='train_model', key='model_type')

    if not run_id:
        raise ValueError("No run_id received from train_model via XCom")

    # ── Load test data ────────────────────────────────────
    df_test = pd.read_csv(test_data_path)
    X_test = df_test.drop(columns=['Survived'])
    y_test = df_test['Survived']
    print(f"✓ Loaded test data: {X_test.shape}")

    # ── Load model from MLflow ────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ Loaded model from MLflow run: {run_id}")

    # ── Make predictions ──────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Compute metrics ───────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"\n── Model Evaluation Metrics ──")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # ── Log all metrics to MLflow ─────────────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        print(f"\n✓ All metrics logged to MLflow run: {run_id}")

    # ── Push accuracy via XCom for branching ─────────────
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='run_id', value=run_id)
    print(f"\n✓ XCom pushed: accuracy = {accuracy:.4f}")

def branch_decision_fn(**context):
    ti = context['ti']

    # ── Pull accuracy from XCom ───────────────────────────
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')

    if accuracy is None:
        raise ValueError("No accuracy received from evaluate_model via XCom")

    print(f"── Branch Decision ──")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Threshold: 0.80")

    # ── Decision logic ────────────────────────────────────
    if accuracy >= 0.80:
        print(f"\n✓ Accuracy {accuracy:.4f} >= 0.80 → Registering model")
        return 'register_model'
    else:
        print(f"\n✗ Accuracy {accuracy:.4f} < 0.80 → Rejecting model")
        return 'reject_model'

def reject_model_fn(**context):
    ti = context['ti']

    # ── Pull accuracy from XCom ───────────────────────────
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')

    # ── Log rejection reason ──────────────────────────────
    reason = f"Model rejected — accuracy {accuracy:.4f} is below 0.80 threshold"
    print(f"\n✗ {reason}")

    # ── Log rejection to MLflow ───────────────────────────
    run_id = ti.xcom_pull(task_ids='evaluate_model', key='run_id')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("status", "rejected")
        mlflow.set_tag("rejection_reason", reason)
        print(f"✓ Rejection reason logged to MLflow")

def register_model_fn(**context):
    ti = context['ti']

    # ── Pull run_id from XCom ─────────────────────────────
    run_id = ti.xcom_pull(task_ids='evaluate_model', key='run_id')
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')
    model_type = ti.xcom_pull(task_ids='train_model', key='model_type')

    if not run_id:
        raise ValueError("No run_id received from evaluate_model via XCom")

    print(f"── Model Registration ──")
    print(f"  Run ID:     {run_id}")
    print(f"  Model Type: {model_type}")
    print(f"  Accuracy:   {accuracy:.4f}")

    # ── Set MLflow tracking URI ───────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── Register model in MLflow Model Registry ───────────
    model_uri = f"runs:/{run_id}/model"
    model_name = "titanic_survival_model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(f"\n✓ Model registered successfully!")
    print(f"  Name:    {registered_model.name}")
    print(f"  Version: {registered_model.version}")

    # ── Add description and tags to registered model ──────
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    client.update_model_version(
        name=model_name,
        version=registered_model.version,
        description=f"Titanic survival model — {model_type} with accuracy {accuracy:.4f}"
    )

    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="status",
        value="approved"
    )

    client.set_model_version_tag(
        name=model_name,
        version=registered_model.version,
        key="accuracy",
        value=str(round(accuracy, 4))
    )

    print(f"\n✓ Model description and tags set")
    print(f"  Description: Titanic survival model — {model_type}")
    print(f"  Tag status:  approved")
    print(f"  Tag accuracy: {accuracy:.4f}")

    # ── Log registration status back to MLflow run ────────
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("status", "registered")
        mlflow.set_tag("registered_model_name", model_name)
        mlflow.set_tag("registered_model_version", str(registered_model.version))
        print(f"\n✓ Registration tags logged to MLflow run")

with DAG(
    dag_id='titanic_mlops_pipeline',
    default_args=default_args,
    description='End-to-end Titanic survival ML pipeline',
    schedule_interval=None,                # Run manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'titanic', 'mlflow'],
) as dag:

    # ── TASK 2: Data Ingestion ──────────────────
    ingest_data = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_fn,
        provide_context=True,
    )

    # ── TASK 3: Data Validation ─────────────────
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_fn,
        provide_context=True,
        retries=2,                            # Retry up to 2 times on failure
        retry_delay=timedelta(seconds=30),
    )

    # ── TASK 4a: Handle Missing Values ──────────
    handle_missing = PythonOperator(
        task_id='handle_missing_values',
        python_callable=handle_missing_fn,
        provide_context=True,
    )

    # ── TASK 4b: Feature Engineering ────────────
    feature_engineering = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering_fn,
        provide_context=True,
    )

    # ── TASK 5: Data Encoding ───────────────────
    encode_data = PythonOperator(
        task_id='encode_data',
        python_callable=encode_data_fn,
        provide_context=True,
    )

    # ── TASK 6: Model Training ──────────────────
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_fn,
        provide_context=True,
    )
    # ── TASK 7: Model Evaluation ────────────────
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_fn,
        provide_context=True,
    )

    # ── TASK 8: Branching Logic ─────────────────
    branch_decision = BranchPythonOperator(
        task_id='branch_decision',
        python_callable=branch_decision_fn,
        provide_context=True,
    )

    # ── TASK 9a: Register Model ─────────────────
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model_fn,
        provide_context=True,
    )

    # ── TASK 9b: Reject Model ───────────────────
    reject_model = PythonOperator(
        task_id='reject_model',
        python_callable=reject_model_fn,
        provide_context=True,
    )

    # ── End marker ──────────────────────────────
    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    # ──────────────────────────────────────────────
    # TASK DEPENDENCIES
    # ──────────────────────────────────────────────
    ingest_data >> validate_data

    # Parallel tasks
    validate_data >> [handle_missing, feature_engineering]

    # Both parallel tasks feed into encode_data
    [handle_missing, feature_engineering] >> encode_data

    encode_data >> train_model >> evaluate_model >> branch_decision

    # Branching paths
    branch_decision >> register_model
    branch_decision >> reject_model

    # Both paths converge at end
    [register_model, reject_model] >> end
