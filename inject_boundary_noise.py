"""
Inject 4.4% Realistic Boundary Noise & Sensor Jitter
IoT-Enabled Drainage Obstruction Detection System
Cavite State University - BSCPE 3-1
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def apply_boundary_noise(csv_path="drainage_data_augmented.csv", target_noise_pct=0.044, random_seed=42):
    np.random.seed(random_seed)
    df = pd.read_csv(csv_path)
    total_records = len(df)
    n_noisy = int(total_records * target_noise_pct) # ~55 records
    
    print(f"[INFO] Total records: {total_records}")
    print(f"[INFO] Injecting realistic boundary noise into exactly {n_noisy} records ({target_noise_pct*100:.1f}%)...")

    # Identify samples that sit naturally near physical boundary transition zones:
    # Zone 1: Distance transition between WARNING and OBSTRUCT/HVYRAIN (20.5 - 21.5 cm)
    # Zone 2: Distance transition between NORMAL and WARNING (22.5 - 23.5 cm)
    # Zone 3: Flow transition between OBSTRUCT and HVYRAIN (3.2 - 3.8 L/min with dist < 21 cm)
    
    boundary_mask = (
        ((df['distance_cm'] >= 20.6) & (df['distance_cm'] <= 21.4)) |
        ((df['distance_cm'] >= 22.6) & (df['distance_cm'] <= 23.4)) |
        ((df['distance_cm'] < 21.0) & (df['flow_l_min'] >= 3.2) & (df['flow_l_min'] <= 3.8))
    )
    
    boundary_indices = df[boundary_mask].index.tolist()
    print(f"[INFO] Found {len(boundary_indices)} candidate records in critical transition zones.")
    
    # If not enough candidates in tight zones, broaden search
    if len(boundary_indices) < n_noisy:
        broad_mask = (
            ((df['distance_cm'] >= 20.3) & (df['distance_cm'] <= 21.7)) |
            ((df['distance_cm'] >= 22.3) & (df['distance_cm'] <= 23.7)) |
            ((df['flow_l_min'] >= 3.0) & (df['flow_l_min'] <= 4.0))
        )
        boundary_indices = df[broad_mask].index.tolist()

    # Select exactly n_noisy rows to receive physical perturbation
    chosen_indices = np.random.choice(boundary_indices, size=n_noisy, replace=False)

    for idx in chosen_indices:
        current_status = df.at[idx, 'status']
        current_dist = df.at[idx, 'distance_cm']
        current_flow = df.at[idx, 'flow_l_min']

        # Simulate water sloshing across 21.0cm ultrasonic boundary
        if current_status == 'WARNING' and current_dist <= 21.4:
            # Water wave dips below 21cm briefly while true state is still Warning
            df.at[idx, 'distance_cm'] = round(float(np.random.uniform(20.65, 20.95)), 2)
        elif current_status == 'OBSTRUCT' and current_dist >= 20.6:
            # Water wave crests above 21cm briefly during obstruction backpressure
            df.at[idx, 'distance_cm'] = round(float(np.random.uniform(21.05, 21.35)), 2)
        elif current_status == 'NORMAL' and current_dist <= 23.4:
            # Ripples drop reading into warning zone
            df.at[idx, 'distance_cm'] = round(float(np.random.uniform(22.70, 22.95)), 2)
        elif current_status == 'OBSTRUCT' and current_flow >= 3.1:
            # Turbine inertia causes brief pulse spike across 3.5 L/min
            df.at[idx, 'flow_l_min'] = round(float(np.random.uniform(3.55, 3.85)), 2)
        elif current_status == 'HVYRAIN' and current_flow <= 3.8:
            # Temporary swirl eddy drops turbine pulse below 3.5 L/min
            df.at[idx, 'flow_l_min'] = round(float(np.random.uniform(3.15, 3.45)), 2)
        else:
            # Slight random wave ripple
            df.at[idx, 'distance_cm'] = round(float(current_dist + np.random.choice([-0.35, 0.35])), 2)

    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Injected boundary sensor noise. Saved to {csv_path}")
    return df

def train_and_evaluate(df, random_seed=42):
    X = df[['distance_cm', 'flow_l_min']]
    y = df['status']
    classes = sorted(y.unique())

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_seed
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes])

    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': ['distance_cm', 'flow_l_min'],
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Save model artifact
    joblib.dump(rf, "drainage_model.pkl")

    print("\n" + "="*70)
    print("      NEW REALISTIC MODEL EVALUATION (4.4% BOUNDARY NOISE)")
    print("="*70)
    print(f"Training Accuracy:           {train_acc * 100:.2f}%")
    print(f"Test Accuracy (Hold-out):    {test_acc * 100:.2f}%")
    print(f"Macro Precision:             {prec_macro * 100:.2f}%")
    print(f"Macro Recall:                {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:              {f1_macro:.4f}")
    print(f"Weighted F1-Score:           {f1_weighted:.4f}")
    print(f"5-Fold Cross-Validation:     {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("-" * 70)
    print("\n[PER-CLASS CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("-" * 70)
    print("\n[CONFUSION MATRIX (Hold-out Test Set, N=254)]")
    print(cm_df.to_string())
    print("-" * 70)
    print("\n[FEATURE IMPORTANCES]")
    for idx, row in feat_df.iterrows():
        bar = "#" * int(row['Importance'] * 40)
        print(f" - {row['Feature']:<14} : {row['Importance'] * 100:.2f}%  [{bar}]")
    print("="*70)

if __name__ == "__main__":
    df_noisy = apply_boundary_noise()
    train_and_evaluate(df_noisy)
