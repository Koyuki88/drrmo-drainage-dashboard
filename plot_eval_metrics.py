"""
Model Evaluation Plot & Visualization Generator
IoT-Enabled Drainage Obstruction Detection System
Cavite State University - BSCPE 3-1

Generates high-resolution publication-quality evaluation figures:
1. Confusion Matrix Heatmap
2. Per-Class Metrics Comparison Bar Chart (Precision, Recall, F1)
3. 2D Decision Boundary Map (Distance vs Flow Rate)
4. Feature Importance Contribution Chart
5. Master 4-Panel Evaluation Dashboard (model_evaluation_dashboard.png)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def generate_evaluation_plots(
    data_path="drainage_data_augmented.csv",
    model_path="drainage_model.pkl",
    output_dir="."
):
    # Set clean engineering aesthetic
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#404040'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    # Load data and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    X = df[['distance_cm', 'flow_l_min']]
    y = df['status']
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)

    # -------------------------------------------------------------------------
    # MASTER 4-PANEL DASHBOARD FIGURE (300 DPI)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    fig.patch.set_facecolor('#fdfcfb')

    # Color palette
    color_map = {
        'NORMAL': '#107c41',    # Green
        'WARNING': '#d97706',   # Amber
        'OBSTRUCT': '#b02631',  # Crimson Red
        'HVYRAIN': '#2563eb'    # Blue
    }

    # -------------------------------------------------------------------------
    # Panel 1: Confusion Matrix Heatmap
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#ffffff')
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds, alpha=0.85)
    ax1.set_title("A. Confusion Matrix (Test Set, N=254, Acc=97.24%)", fontsize=12, fontweight='bold', pad=12, color='#181413')
    
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=9, fontweight='semibold')
    ax1.set_yticklabels(classes, fontsize=9, fontweight='semibold')
    ax1.set_xlabel("Predicted Class", fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_ylabel("True / Empirical Class", fontsize=10, fontweight='bold', labelpad=8)

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > thresh else "#181413"
            ax1.text(j, i, f"{val}\n({val/cm[i].sum()*100:.1f}%)" if cm[i].sum() > 0 else f"{val}",
                     ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # -------------------------------------------------------------------------
    # Panel 2: Per-Class Evaluation Metrics (Precision, Recall, F1)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, zorder=0)
    
    x = np.arange(len(classes))
    width = 0.25

    rects1 = ax2.bar(x - width, prec * 100, width, label='Precision', color='#385065', zorder=3)
    rects2 = ax2.bar(x, rec * 100, width, label='Recall', color='#b02631', zorder=3)
    rects3 = ax2.bar(x + width, f1 * 100, width, label='F1-Score', color='#501152', zorder=3)

    ax2.set_title("B. Per-Class Performance Breakdown", fontsize=12, fontweight='bold', pad=12, color='#181413')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9, fontweight='semibold')
    ax2.set_ylabel("Score (%)", fontsize=10, fontweight='bold')
    ax2.set_ylim(80, 105)
    ax2.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

    # Add value labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 2 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 3: Feature Importance
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#ffffff')
    ax3.grid(True, axis='x', zorder=0)

    features = ['distance_cm (Water Level)', 'flow_l_min (Flow Velocity)']
    importances = model.feature_importances_ * 100
    y_pos = np.arange(len(features))

    bars = ax3.barh(y_pos, importances, color=['#b02631', '#385065'], height=0.45, zorder=3)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(features, fontsize=9, fontweight='semibold')
    ax3.set_xlabel("Relative Importance Weight (%)", fontsize=10, fontweight='bold')
    ax3.set_title("C. Random Forest Feature Contribution Split", fontsize=12, fontweight='bold', pad=12, color='#181413')
    ax3.set_xlim(0, 75)

    for bar, val in zip(bars, importances):
        ax3.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", 
                 va='center', fontsize=10, fontweight='bold', color='#181413')

    # -------------------------------------------------------------------------
    # Panel 4: 2D Decision Boundary Map with Test Points
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#ffffff')
    
    # Meshgrid across feature space
    x_min, x_max = X['distance_cm'].min() - 0.5, X['distance_cm'].max() + 0.5
    y_min, y_max = X['flow_l_min'].min() - 0.5, X['flow_l_min'].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Predict over meshgrid
    Z_raw = model.predict(np.c_[xx.ravel(), yy.ravel()])
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    Z = np.array([class_to_int[z] for z in Z_raw]).reshape(xx.shape)

    # Shaded contour regions
    light_colors = ['#dbeafe', '#dcfce7', '#fee2e2', '#fef3c7'] # HVYRAIN, NORMAL, OBSTRUCT, WARNING
    cmap_light = ListedColormap(light_colors)
    ax4.contourf(xx, yy, Z, alpha=0.55, cmap=cmap_light)

    # Scatter actual test points on top
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

    # Draw hardware thresholds
    ax4.axvline(x=21.0, color='#b02631', linestyle='--', linewidth=1.2, label='Threshold 21.0cm')
    ax4.axvline(x=23.0, color='#107c41', linestyle='--', linewidth=1.2, label='Threshold 23.0cm')
    ax4.axhline(y=3.5, color='#d97706', linestyle='--', linewidth=1.2, label='Threshold 3.5 L/min')

    ax4.set_title("D. 2D Decision Boundaries & Test Data Ingress", fontsize=12, fontweight='bold', pad=12, color='#181413')
    ax4.set_xlabel("Distance from Culvert Ceiling (cm) [Inverted Water Level]", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Water Flow Rate (L/min)", fontsize=10, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_evaluation_dashboard.png")
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Master Evaluation Dashboard saved to: {output_path}")

if __name__ == "__main__":
    generate_evaluation_plots()
