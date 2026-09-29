"""
train_model.py - Machine Learning Pipeline for IoT Drainage Obstruction Detection

This script trains a Random Forest Classifier to predict drainage operational
conditions ('NORMAL', 'WARNING', 'HVYRAIN', 'OBSTRUCT') using real-time
telemetry: ultrasonic distance to water surface ('distance_cm') and flow rate ('flow_l_min').
The trained estimator is serialized to 'drainage_model.pkl' for use by the
Streamlit dashboard (app.py) and cloud analytics.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Canonical feature ordering and target name
FEATURE_COLUMNS = ['distance_cm', 'flow_l_min']
TARGET_COLUMN = 'status'
EXPECTED_CLASSES = ['HVYRAIN', 'NORMAL', 'OBSTRUCT', 'WARNING']


def parse_args():
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier for IoT Drainage Obstruction Detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="drainage_data.csv",
        help="Path to input CSV dataset (default: drainage_data.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drainage_model.pkl",
        help="Path to save exported model file (default: drainage_model.pkl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-training verification assertions"
    )
    return parser.parse_args()


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV and validate structure and features.
    
    Args:
        filepath: Path to the dataset CSV file.
        
    Returns:
        pd.DataFrame containing validated telemetry data.
        
    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If required columns are missing or dataset is empty.
    """
    if not os.path.exists(filepath):
        # Also check relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

    print(f"[DATA] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    # Validate required columns
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}. Found: {list(df.columns)}")

    # Check and clean null values
    initial_count = len(df)
    df = df.dropna(subset=required_cols).copy()
    cleaned_count = len(df)
    if cleaned_count < initial_count:
        print(f"[DATA] Dropped {initial_count - cleaned_count} rows with null/missing values.")

    # Validate data types
    df['distance_cm'] = pd.to_numeric(df['distance_cm'], errors='coerce')
    df['flow_l_min'] = pd.to_numeric(df['flow_l_min'], errors='coerce')
    df['status'] = df['status'].astype(str).str.strip()

    # Drop any rows that failed numeric conversion
    df = df.dropna(subset=required_cols).copy()

    # Verify target classes
    unique_classes = sorted(df['status'].unique())
    print(f"[DATA] Total valid samples: {len(df)}")
    print(f"[DATA] Target classes present: {unique_classes}")
    print("[DATA] Class distribution:")
    for cls_name, count in df['status'].value_counts().items():
        pct = (count / len(df)) * 100.0
        print(f"       - {cls_name:10s}: {count:5d} ({pct:5.2f}%)")

    return df


def train_model(
    data_path: str = "drainage_data.csv",
    output_path: str = "drainage_model.pkl",
    random_state: int = 42,
    n_estimators: int = 100,
    verify: bool = True
) -> RandomForestClassifier:
    """
    Execute full training pipeline, evaluate metrics, and export model artifact.
    
    Args:
        data_path: Path to dataset CSV.
        output_path: Destination path for serialized .pkl model.
        random_state: Random state seed.
        n_estimators: Number of trees in the Random Forest.
        verify: Whether to perform post-training test predictions.
        
    Returns:
        Trained RandomForestClassifier model instance.
    """
    print("=" * 70)
    print("IoT Drainage Analytics - Machine Learning Pipeline (Random Forest)")
    print("=" * 70)

    # 1. Load and clean data
    df = load_and_validate_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
    print(f"\n[SPLIT] Stratified split: {len(X_train)} train samples (80%), {len(X_test)} test samples (20%)")

    # 3. Model instantiation and training
    print(f"[TRAIN] Training RandomForestClassifier (n_estimators={n_estimators}, class_weight='balanced', random_state={random_state})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("[TRAIN] Model training complete.")

    # 4. Evaluation on test set
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 25 + " EVALUATION METRICS " + "=" * 25)
    print(f"Accuracy:            {acc:.4f} ({acc * 100.0:.2f}%)")
    print(f"Precision (Macro):   {prec_macro:.4f}")
    print(f"Precision (Weighted):{prec_weighted:.4f}")
    print(f"Recall (Macro):      {rec_macro:.4f}")
    print(f"Recall (Weighted):   {rec_weighted:.4f}")
    print(f"F1-Score (Macro):    {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("=" * 70)

    print("\n[METRICS] Detailed Classification Report:")
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print(report)

    print("[METRICS] Confusion Matrix:")
    labels = sorted(list(clf.classes_))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Nicely formatted matrix with headers
    header_str = "               " + "".join([f"{lbl:>10s}" for lbl in labels])
    print(header_str)
    for i, lbl in enumerate(labels):
        row_str = f"  {lbl:12s} " + "".join([f"{cm[i, j]:>10d}" for j in range(len(labels))])
        print(row_str)

    # 5. Export model to pickle via joblib
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(clf, output_path)
    file_size_bytes = os.path.getsize(output_path)
    print(f"\n[EXPORT] Model successfully saved to: {output_path} ({file_size_bytes:,} bytes)")

    # 6. Post-training verification
    if verify:
        verify_model_artifact(output_path)

    print("=" * 70)
    print("Training pipeline finished successfully.")
    print("=" * 70)
    return clf


def verify_model_artifact(model_path: str):
    """
    Verify model artifact by loading from disk and testing representative inputs.
    
    Args:
        model_path: Path to serialized .pkl model.
    """
    print("\n" + "=" * 25 + " MODEL VERIFICATION " + "=" * 25)
    print(f"[VERIFY] Loading model from {model_path} via joblib.load()...")
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict"), "Loaded model does not have 'predict' method."
    assert hasattr(loaded_model, "predict_proba"), "Loaded model does not have 'predict_proba' method."

    # Test cases matching physical domain rules:
    # 1. NORMAL: distance ~24.0 cm (water level low), flow ~0.0 L/min -> 'NORMAL'
    # 2. WARNING: distance ~22.0 cm (water level rising), flow ~0.0 L/min -> 'WARNING'
    # 3. HVYRAIN: distance ~20.3 cm (critical flood level), flow ~5.0 L/min (swift flow) -> 'HVYRAIN'
    # 4. OBSTRUCT: distance ~20.3 cm (critical flood level), flow ~1.0 L/min (restricted flow) -> 'OBSTRUCT'
    test_cases = [
        {"distance": 24.0, "flow": 0.0, "expected": "NORMAL"},
        {"distance": 22.0, "flow": 0.0, "expected": "WARNING"},
        {"distance": 20.3, "flow": 5.0, "expected": "HVYRAIN"},
        {"distance": 20.3, "flow": 1.0, "expected": "OBSTRUCT"},
    ]

    all_passed = True
    print("\n[VERIFY] Executing representative test predictions:")
    for tc in test_cases:
        sample_df = pd.DataFrame(
            [[tc["distance"], tc["flow"]]],
            columns=FEATURE_COLUMNS
        )
        prediction = loaded_model.predict(sample_df)[0]
        probs = loaded_model.predict_proba(sample_df)[0]
        pred_idx = list(loaded_model.classes_).index(prediction)
        confidence = probs[pred_idx] * 100.0

        is_correct = (prediction == tc["expected"])
        status_flag = "PASS" if is_correct else "FAIL"
        if not is_correct:
            all_passed = False

        print(
            f"  [{status_flag}] distance={tc['distance']:5.1f} cm, flow={tc['flow']:4.1f} L/min "
            f"-> Pred: {prediction:8s} (Expected: {tc['expected']:8s}) [Conf: {confidence:5.1f}%]"
        )

    if not all_passed:
        raise RuntimeError("Verification failed: one or more representative test predictions did not match expected output.")

    print("\n[VERIFY] All representative domain tests PASSED successfully!")


if __name__ == "__main__":
    args = parse_args()
    try:
        train_model(
            data_path=args.data,
            output_path=args.output,
            random_state=args.seed,
            n_estimators=args.estimators,
            verify=not args.no_verify
        )
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
