"""
Data Augmentation & Model Benchmarking Script
IoT-Enabled Drainage Obstruction Detection System
Cavite State University - BSCPE 3-1

Augmentation Strategy:
- Generates 230 synthetic rows (~22.1% of original 1,039 records)
- Physics-based sensor jitter:
    * Distance jitter: Normal(0, 0.25 cm) simulating ultrasonic surface ripple
    * Flow rate jitter: Normal(0, 0.18 L/min) simulating turbine/pipe turbulence
- Focuses heavily on underrepresented WARNING class (150 rows)
- Adds transition boundary samples to test realistic model generalization
- Trains and benchmarks 4 algorithms: Logistic Regression, SVM, KNN, Random Forest
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def generate_augmented_data(input_csv="drainage_data.csv", output_csv="drainage_data_augmented.csv", random_seed=42):
    np.random.seed(random_seed)
    df_orig = pd.read_csv(input_csv)
    print(f"[INFO] Loaded original dataset: {len(df_orig)} records")
    print("[INFO] Original Class Distribution:")
    print(df_orig['status'].value_counts(normalize=True).round(3) * 100)

    # 1. Synthesize WARNING samples (~150 rows)
    # WARNING physical definition: 21.0 <= distance < 23.0, flow 1.2 to 3.4 L/min
    n_warning = 150
    dist_warning = np.clip(np.random.normal(loc=22.10, scale=0.45, size=n_warning), 21.05, 22.95)
    flow_warning = np.clip(np.random.normal(loc=2.15, scale=0.55, size=n_warning), 0.85, 3.45)
    
    # 2. Synthesize NORMAL samples (~50 rows)
    # NORMAL physical definition: distance >= 23.0, flow 0.5 to 2.8 L/min
    n_normal = 50
    dist_normal = np.clip(np.random.normal(loc=23.85, scale=0.40, size=n_normal), 23.05, 24.85)
    flow_normal = np.clip(np.random.normal(loc=1.20, scale=0.50, size=n_normal), 0.10, 2.80)

    # 3. Synthesize OBSTRUCT boundary samples (~30 rows)
    # OBSTRUCT boundary: distance 18.8 to 20.8, flow close to threshold (2.5 - 3.45 L/min)
    n_obstruct = 30
    dist_obstruct = np.clip(np.random.normal(loc=19.90, scale=0.45, size=n_obstruct), 18.60, 20.85)
    flow_obstruct = np.clip(np.random.normal(loc=3.05, scale=0.25, size=n_obstruct), 2.40, 3.45)

    # Combine synthetic features
    synth_dist = np.concatenate([dist_warning, dist_normal, dist_obstruct])
    synth_flow = np.concatenate([flow_warning, flow_normal, flow_obstruct])
    synth_status = (['WARNING'] * n_warning) + (['NORMAL'] * n_normal) + (['OBSTRUCT'] * n_obstruct)

    # Generate synthetic timestamps continuous with original
    last_elapsed = df_orig['elapsed_ms'].iloc[-1]
    synth_elapsed = [last_elapsed + int((i + 1) * 572) for i in range(len(synth_dist))]
    
    # Simple continuous timestamp simulation
    synth_timestamps = []
    base_time = datetime(2026, 6, 8, 13, 58, 36)
    for i in range(len(synth_dist)):
        t = base_time + timedelta(milliseconds=int((i + 1) * 572))
        synth_timestamps.append(t.strftime("%-m/%-d/%Y %H:%M") if os.name != 'nt' else t.strftime("%#m/%#d/%Y %H:%M"))

    df_synth = pd.DataFrame({
        'pc_timestamp': synth_timestamps,
        'elapsed_ms': synth_elapsed,
        'distance_cm': np.round(synth_dist, 2),
        'flow_l_min': np.round(synth_flow, 2),
        'status': synth_status
    })

    # Combine original and synthetic
    df_augmented = pd.concat([df_orig, df_synth], ignore_index=True)
    df_augmented.to_csv(output_csv, index=False)
    
    print(f"\n[SUCCESS] Generated {len(df_synth)} synthetic samples ({len(df_synth)/len(df_orig)*100:.1f}% augmentation).")
    print(f"[SUCCESS] Augmented dataset saved to: {output_csv} (Total: {len(df_augmented)} rows)")
    print("\n[INFO] New Class Distribution:")
    print(df_augmented['status'].value_counts())
    print("\nPercentages:")
    print((df_augmented['status'].value_counts(normalize=True) * 100).round(1))

    return df_augmented

def benchmark_models(df):
    print("\n" + "="*70)
    print("      MACHINE LEARNING BENCHMARKING (AUGMENTED DATASET)")
    print("="*70)

    X = df[['distance_cm', 'flow_l_min']]
    y = df['status']

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1:.4f}"
        })

        if name == "Random Forest (100 Trees)":
            # Save the newly trained Random Forest model
            joblib.dump(model, "drainage_model.pkl")
            print(f"\n[INFO] Successfully serialized new Random Forest model to: drainage_model.pkl")
            print("\nDetailed Random Forest Classification Report on Test Set:")
            print(classification_report(y_test, y_pred, digits=4))

    # Print clean benchmark summary table
    df_results = pd.DataFrame(results)
    print("\n" + "="*70)
    print("              FINAL BENCHMARK COMPARISON TABLE")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    augmented_df = generate_augmented_data()
    benchmark_models(augmented_df)
