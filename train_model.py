"""
train_model.py - Machine Learning Pipeline for IoT Drainage Obstruction Detection

This script trains a Random Forest Classifier to predict drainage operational
conditions ('NORMAL', 'WARNING', 'HVYRAIN', 'OBSTRUCT') using real-time
telemetry: ultrasonic distance to water surface ('distance_cm') and flow rate ('flow_l_min').
The trained estimator is serialized to 'drainage_model.pkl'.

Features:
- Automatically defaults to 'drainage_data_augmented.csv' if present
- Generates 4 SEPARATE high-resolution (300 DPI) publication evaluation plots:
    1. plot_confusion_matrix.png
    2. plot_per_class_metrics.png
    3. plot_feature_importance.png
    4. plot_decision_boundaries.png
    5. model_evaluation_dashboard.png (bonus 4-panel combined)
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Configure non-interactive backend for reliable CLI plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# Canonical feature ordering and target name
FEATURE_COLUMNS = ['distance_cm', 'flow_l_min']
TARGET_COLUMN = 'status'
EXPECTED_CLASSES = ['HVYRAIN', 'NORMAL', 'OBSTRUCT', 'WARNING']


def get_default_dataset():
    """Pick augmented dataset if it exists, otherwise fall back to raw CSV."""
    if os.path.exists("drainage_data_augmented.csv"):
        return "drainage_data_augmented.csv"
    return "drainage_data.csv"


def parse_args():
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier for IoT Drainage Obstruction Detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=get_default_dataset(),
        help=f"Path to input CSV dataset (default: {get_default_dataset()})"
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
        "--no-plots",
        action="store_true",
        help="Skip generating evaluation plot images"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-training verification assertions"
    )
    return parser.parse_args()


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV and validate structure and features."""
    if not os.path.exists(filepath):
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

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}. Found: {list(df.columns)}")

    # Clean null values
    df = df.dropna(subset=required_cols).copy()
    df['distance_cm'] = pd.to_numeric(df['distance_cm'], errors='coerce')
    df['flow_l_min'] = pd.to_numeric(df['flow_l_min'], errors='coerce')
    df['status'] = df['status'].astype(str).str.strip()
    df = df.dropna(subset=required_cols).copy()

    unique_classes = sorted(df['status'].unique())
    print(f"[DATA] Total valid samples: {len(df)}")
    print(f"[DATA] Target classes present: {unique_classes}")
    print("[DATA] Class distribution:")
    for cls_name, count in df['status'].value_counts().items():
        pct = (count / len(df)) * 100.0
        print(f"       - {cls_name:10s}: {count:5d} ({pct:5.2f}%)")

    return df


def generate_evaluation_plots(clf, X_test, y_test, X, y, output_dir="."):
    """
    Generate SEPARATE individual high-resolution (300 DPI) evaluation plots.
    """
    print("\n" + "=" * 25 + " GENERATING EVALUATION PLOTS " + "=" * 25)
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#404040'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    classes = sorted(list(clf.classes_))
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    color_map = {
        'NORMAL': '#107c41',    # Green
        'WARNING': '#d97706',   # Amber
        'OBSTRUCT': '#b02631',  # Crimson Red
        'HVYRAIN': '#2563eb'    # Blue
    }

    # -------------------------------------------------------------------------
    # PLOT 1: Confusion Matrix Heatmap (Separate PNG)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)
    fig1.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds, alpha=0.88)
    ax1.set_title(f"Confusion Matrix (Test Set N={len(y_test)}, Acc={acc*100:.2f}%)", fontsize=11, fontweight='bold', pad=12)
    
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=9, fontweight='semibold')
    ax1.set_yticklabels(classes, fontsize=9, fontweight='semibold')
    ax1.set_xlabel("Predicted Class", fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_ylabel("True / Empirical Class", fontsize=10, fontweight='bold', labelpad=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > thresh else "#181413"
            pct_str = f"{val/cm[i].sum()*100:.1f}%" if cm[i].sum() > 0 else "0%"
            ax1.text(j, i, f"{val}\n({pct_str})", ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "plot_confusion_matrix.png")
    fig1.savefig(p1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f" [PLOT 1/4] Saved Confusion Matrix       -> {p1_path}")

    # -------------------------------------------------------------------------
    # PLOT 2: Per-Class Performance Bar Chart (Separate PNG)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=300)
    fig2.patch.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, zorder=0)

    x = np.arange(len(classes))
    width = 0.26
    r1 = ax2.bar(x - width, prec * 100, width, label='Precision', color='#385065', zorder=3)
    r2 = ax2.bar(x, rec * 100, width, label='Recall', color='#b02631', zorder=3)
    r3 = ax2.bar(x + width, f1 * 100, width, label='F1-Score', color='#501152', zorder=3)

    ax2.set_title("Per-Class Classification Metrics Breakdown", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9, fontweight='semibold')
    ax2.set_ylabel("Metric Score (%)", fontsize=10, fontweight='bold')
    ax2.set_ylim(80, 105)
    ax2.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    p2_path = os.path.join(output_dir, "plot_per_class_metrics.png")
    fig2.savefig(p2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f" [PLOT 2/4] Saved Per-Class Metrics     -> {p2_path}")

    # -------------------------------------------------------------------------
    # PLOT 3: Feature Importance Bar Chart (Separate PNG)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 4.5), dpi=300)
    fig3.patch.set_facecolor('#ffffff')
    ax3.set_facecolor('#ffffff')
    ax3.grid(True, axis='x', zorder=0)

    feat_names = ['distance_cm (Water Depth)', 'flow_l_min (Flow Velocity)']
    feat_vals = clf.feature_importances_ * 100
    y_pos = np.arange(len(feat_names))

    bars = ax3.barh(y_pos, feat_vals, color=['#b02631', '#385065'], height=0.42, zorder=3)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feat_names, fontsize=9, fontweight='semibold')
    ax3.set_xlabel("Relative Importance Weight (%)", fontsize=10, fontweight='bold')
    ax3.set_title("Random Forest Feature Contribution Split", fontsize=11, fontweight='bold', pad=12)
    ax3.set_xlim(0, 75)

    for bar, val in zip(bars, feat_vals):
        ax3.text(val + 1.2, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", 
                 va='center', fontsize=9.5, fontweight='bold', color='#181413')

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "plot_feature_importance.png")
    fig3.savefig(p3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f" [PLOT 3/4] Saved Feature Importance    -> {p3_path}")

    # -------------------------------------------------------------------------
    # PLOT 4: 2D Decision Boundaries with Test Ingress (Separate PNG)
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=300)
    fig4.patch.set_facecolor('#ffffff')
    ax4.set_facecolor('#ffffff')

    x_min, x_max = X['distance_cm'].min() - 0.5, X['distance_cm'].max() + 0.5
    y_min, y_max = X['flow_l_min'].min() - 0.5, X['flow_l_min'].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z_raw = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    Z = np.array([class_to_int[z] for z in Z_raw]).reshape(xx.shape)

    light_colors = ['#dbeafe', '#dcfce7', '#fee2e2', '#fef3c7']
    cmap_light = ListedColormap(light_colors)
    ax4.contourf(xx, yy, Z, alpha=0.55, cmap=cmap_light)

    for cls in classes:
        mask = (y_test == cls)
        ax4.scatter(
            X_test.loc[mask, 'distance_cm'],
            X_test.loc[mask, 'flow_l_min'],
            label=f"Test {cls}",
            color=color_map[cls],
            edgecolor='#ffffff',
            linewidth=0.5,
            s=40,
            alpha=0.9
        )

    ax4.axvline(x=21.0, color='#b02631', linestyle='--', linewidth=1.2, label='Threshold 21.0cm')
    ax4.axvline(x=23.0, color='#107c41', linestyle='--', linewidth=1.2, label='Threshold 23.0cm')
    ax4.axhline(y=3.5, color='#d97706', linestyle='--', linewidth=1.2, label='Threshold 3.5 L/min')

    ax4.set_title("2D Decision Boundary Topography & Test Data Overlay", fontsize=11, fontweight='bold', pad=12)
    ax4.set_xlabel("Distance from Sensor to Water Surface (cm)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Water Flow Rate (L/min)", fontsize=10, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

    plt.tight_layout()
    p4_path = os.path.join(output_dir, "plot_decision_boundaries.png")
    fig4.savefig(p4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f" [PLOT 4/4] Saved Decision Boundaries   -> {p4_path}")

    # -------------------------------------------------------------------------
    # BONUS: Combined 4-Panel Master Dashboard
    # -------------------------------------------------------------------------
    fig_all, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    fig_all.patch.set_facecolor('#fdfcfb')
    
    # 1. CM
    ax_cm = axes[0, 0]
    ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds, alpha=0.88)
    ax_cm.set_title(f"A. Confusion Matrix (Acc={acc*100:.2f}%)", fontsize=11, fontweight='bold')
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(classes, fontsize=8.5, fontweight='semibold')
    ax_cm.set_yticklabels(classes, fontsize=8.5, fontweight='semibold')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            tc = "white" if val > thresh else "#181413"
            ax_cm.text(j, i, f"{val}", ha="center", va="center", color=tc, fontsize=9, fontweight='bold')

    # 2. Metrics
    ax_m = axes[0, 1]
    ax_m.grid(True, zorder=0)
    ax_m.bar(x - width, prec * 100, width, label='Precision', color='#385065', zorder=3)
    ax_m.bar(x, rec * 100, width, label='Recall', color='#b02631', zorder=3)
    ax_m.bar(x + width, f1 * 100, width, label='F1', color='#501152', zorder=3)
    ax_m.set_title("B. Classification Metrics", fontsize=11, fontweight='bold')
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(classes, fontsize=8.5, fontweight='semibold')
    ax_m.set_ylim(80, 105)
    ax_m.legend(loc='lower right', fontsize=8)

    # 3. Features
    ax_f = axes[1, 0]
    ax_f.grid(True, axis='x', zorder=0)
    bars_f = ax_f.barh(y_pos, feat_vals, color=['#b02631', '#385065'], height=0.42, zorder=3)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(feat_names, fontsize=8.5, fontweight='semibold')
    ax_f.set_title("C. Feature Contributions", fontsize=11, fontweight='bold')
    ax_f.set_xlim(0, 75)
    for bar, val in zip(bars_f, feat_vals):
        ax_f.text(val + 1.2, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va='center', fontsize=9, fontweight='bold')

    # 4. Contours
    ax_b = axes[1, 1]
    ax_b.contourf(xx, yy, Z, alpha=0.55, cmap=cmap_light)
    for cls in classes:
        mask = (y_test == cls)
        ax_b.scatter(X_test.loc[mask, 'distance_cm'], X_test.loc[mask, 'flow_l_min'], color=color_map[cls], s=25, alpha=0.9)
    ax_b.set_title("D. Decision Boundary Map", fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Distance (cm)", fontsize=9, fontweight='bold')
    ax_b.set_ylabel("Flow (L/min)", fontsize=9, fontweight='bold')

    plt.tight_layout()
    master_path = os.path.join(output_dir, "model_evaluation_dashboard.png")
    fig_all.savefig(master_path, dpi=300, bbox_inches='tight')
    plt.close(fig_all)
    print(f" [BONUS]   Saved Master 4-Panel Dashboard -> {master_path}")
    print("=" * 70)


def train_model(
    data_path: str = None,
    output_path: str = "drainage_model.pkl",
    random_state: int = 42,
    n_estimators: int = 100,
    generate_plots: bool = True,
    verify: bool = True
) -> RandomForestClassifier:
    """Execute full training pipeline, evaluate metrics, and export model artifact."""
    if data_path is None:
        data_path = get_default_dataset()

    print("=" * 70)
    print("IoT Drainage Analytics - Machine Learning Pipeline (Random Forest)")
    print("=" * 70)

    # 1. Load data
    df = load_and_validate_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"\n[SPLIT] Stratified split: {len(X_train)} train samples (80%), {len(X_test)} test samples (20%)")

    # 3. Model training
    print(f"[TRAIN] Training RandomForestClassifier (n_estimators={n_estimators}, class_weight='balanced', random_state={random_state})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("[TRAIN] Model training complete.")

    # 4. Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')

    print("\n" + "=" * 25 + " EVALUATION METRICS " + "=" * 25)
    print(f"Accuracy (Hold-out):   {acc:.4f} ({acc * 100.0:.2f}%)")
    print(f"5-Fold Cross-Val:      {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print(f"Precision (Macro):     {prec_macro:.4f} ({prec_macro * 100.0:.2f}%)")
    print(f"Recall (Macro):        {rec_macro:.4f} ({rec_macro * 100.0:.2f}%)")
    print(f"F1-Score (Macro):      {f1_macro:.4f}")
    print(f"F1-Score (Weighted):   {f1_weighted:.4f}")
    print("=" * 70)

    print("\n[METRICS] Detailed Classification Report:")
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print(report)

    print("[METRICS] Confusion Matrix:")
    labels = sorted(list(clf.classes_))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header_str = "               " + "".join([f"{lbl:>10s}" for lbl in labels])
    print(header_str)
    for i, lbl in enumerate(labels):
        row_str = f"  {lbl:12s} " + "".join([f"{cm[i, j]:>10d}" for j in range(len(labels))])
        print(row_str)

    # 5. Export model
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(clf, output_path)
    file_size_bytes = os.path.getsize(output_path)
    print(f"\n[EXPORT] Model successfully saved to: {output_path} ({file_size_bytes:,} bytes)")

    # 6. Generate SEPARATE evaluation plots
    if generate_plots:
        generate_evaluation_plots(clf, X_test, y_test, X, y, output_dir=".")

    # 7. Post-training verification
    if verify:
        verify_model_artifact(output_path)

    print("=" * 70)
    print("Training pipeline and plot generation finished successfully.")
    print("=" * 70)
    return clf


def verify_model_artifact(model_path: str):
    """Verify model artifact by loading from disk and testing representative inputs."""
    print("\n" + "=" * 25 + " MODEL VERIFICATION " + "=" * 25)
    print(f"[VERIFY] Loading model from {model_path} via joblib.load()...")
    loaded_model = joblib.load(model_path)

    test_cases = [
        {"distance": 24.0, "flow": 0.0, "expected": "NORMAL"},
        {"distance": 22.0, "flow": 2.0, "expected": "WARNING"},
        {"distance": 19.5, "flow": 5.0, "expected": "HVYRAIN"},
        {"distance": 19.5, "flow": 1.0, "expected": "OBSTRUCT"},
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
            generate_plots=not args.no_plots,
            verify=not args.no_verify
        )
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
