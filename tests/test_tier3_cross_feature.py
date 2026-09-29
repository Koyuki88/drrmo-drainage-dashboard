"""
test_tier3_cross_feature.py - Tier 3: Cross-Feature Combinations & Pairwise Contract Alignment

Verifies consistency and interface contracts across components:
- Telemetry schema alignment across Azure Guide (R1), ML Training (R2), and Streamlit (R3)
- Concordance between ML predictions and DRRMO alert severity mapping
- Physical water depth invariant (25.0 - distance_cm) across Edge, Cloud, and Dashboard
- High concordance (>98%) between firmware rule-based logic and ML classifier
- Dataset integrity and schema compatibility
"""

import os
import json
import re
import pytest
import pandas as pd
import numpy as np

import train_model
import app


class TestTier3CrossFeatureContracts:
    """Pairwise interface contracts and cross-module schema alignment."""

    def test_schema_alignment_feature_columns_across_modules(self, file_paths):
        """Case 1: Ensure FEATURE_COLUMNS matches exactly between train_model.py, app.py, and Azure Guide schema."""
        train_features = train_model.FEATURE_COLUMNS
        app_features = app.FEATURE_COLUMNS

        # Check exact equality between ML pipeline and Dashboard
        assert train_features == ["distance_cm", "flow_l_min"], f"Unexpected train_model features: {train_features}"
        assert app_features == train_features, f"Feature mismatch: train_model={train_features} vs app={app_features}"

        # Verify Azure Integration Guide defines both features in JSON schema
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            guide_content = f.read()

        assert '"distance_cm"' in guide_content, "distance_cm missing from Azure Guide JSON schema."
        assert '"flow_l_min"' in guide_content, "flow_l_min missing from Azure Guide JSON schema."

    def test_schema_alignment_target_classes_across_modules(self, file_paths):
        """Case 2: Ensure target classes match across train_model, app.py, DRRMO matrix, and Azure Guide."""
        canonical_classes = {"HVYRAIN", "NORMAL", "OBSTRUCT", "WARNING"}

        # 1. train_model EXPECTED_CLASSES
        assert set(train_model.EXPECTED_CLASSES) == canonical_classes

        # 2. app.py CLASS_LABELS
        assert set(app.CLASS_LABELS) == canonical_classes

        # 3. app.py DRRMO_ALERT_MATRIX keys
        assert set(app.DRRMO_ALERT_MATRIX.keys()) == canonical_classes

        # 4. Azure Guide enum definition
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            guide_content = f.read()

        for cls_name in canonical_classes:
            assert f'"{cls_name}"' in guide_content, f"Target class '{cls_name}' missing from Azure Guide schema enum."

    def test_physical_water_depth_invariant_cross_check(self, file_paths):
        """Case 3: Verify the physical depth formula 'water_level_cm = 25.0 - distance_cm' is consistent across all files."""
        # 1. Check in app.py constants
        assert app.CANAL_BED_DISTANCE_CM == 25.00

        # 2. Check in AZURE_INTEGRATION_GUIDE.md
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            guide_content = f.read()
        assert "25.0 - distance_cm" in guide_content or "25.0 - clamped_dist" in guide_content, \
            "Depth calculation formula missing from Azure Integration Guide."

        # 3. Assert invariant computation for sample points
        test_distances = [25.0, 23.0, 21.0, 20.0, 15.0]
        for dist in test_distances:
            expected_depth = round(max(0.0, 25.0 - dist), 2)
            calculated_depth = round(max(0.0, app.CANAL_BED_DISTANCE_CM - dist), 2)
            assert expected_depth == calculated_depth

    def test_model_predictions_map_1_to_1_to_drrmo_alert_levels(self, loaded_model):
        """Case 4: Verify all model classes map directly to DRRMO alert matrix with comprehensive metadata."""
        required_alert_fields = [
            "level_title", "badge", "banner_class", "color", "bg_color",
            "buzzer_state", "buzzer_timing", "strobe_state",
            "description", "action_protocol", "risk_level"
        ]

        for cls_label in loaded_model.classes_:
            assert cls_label in app.DRRMO_ALERT_MATRIX, f"Model class '{cls_label}' has no corresponding DRRMO alert entry."
            alert_entry = app.DRRMO_ALERT_MATRIX[cls_label]
            for field in required_alert_fields:
                assert field in alert_entry, f"DRRMO alert entry for '{cls_label}' missing required metadata field '{field}'."
                assert len(str(alert_entry[field])) > 0, f"Field '{field}' for '{cls_label}' is empty."

    def test_firmware_to_ml_model_high_concordance_on_dataset(self, loaded_model, raw_dataset):
        """Case 5: Verify >98% prediction concordance between firmware threshold rule engine and trained ML model on historical dataset."""
        df_clean = raw_dataset.dropna(subset=["distance_cm", "flow_l_min", "status"]).copy()
        features_df = df_clean[["distance_cm", "flow_l_min"]]

        # Machine Learning Predictions
        ml_predictions = loaded_model.predict(features_df)

        # Firmware Rule Predictions
        firmware_predictions = []
        for _, row in features_df.iterrows():
            fw_status, _, _ = app.rule_based_predict(row["distance_cm"], row["flow_l_min"])
            firmware_predictions.append(fw_status)

        matches = (ml_predictions == np.array(firmware_predictions)).sum()
        total = len(df_clean)
        concordance_rate = (matches / total) * 100.0

        assert concordance_rate >= 98.0, (
            f"Concordance between ML model and firmware rules is too low: {concordance_rate:.2f}% "
            f"({matches}/{total} matched). Expected >= 98.0%."
        )

    def test_raw_dataset_telemetry_schema_integrity(self, raw_dataset):
        """Case 6: Validate raw CSV dataset strictly conforms to required types, headers, and value domains."""
        expected_cols = ["elapsed_ms", "distance_cm", "flow_l_min", "status"]
        for col in expected_cols:
            assert col in raw_dataset.columns, f"Missing required column '{col}' in drainage_data.csv"

        # Check value constraints
        assert (raw_dataset["distance_cm"] >= 0).all(), "Negative distance values found in dataset."
        assert (raw_dataset["distance_cm"] <= 50.0).all(), "Unrealistic distance values (> 50cm) found."
        assert (raw_dataset["flow_l_min"] >= 0).all(), "Negative flow rates found in dataset."

        # Unique classes present in dataset must be subset of expected classes
        unique_classes = set(raw_dataset["status"].dropna().unique())
        expected_classes = {"NORMAL", "WARNING", "HVYRAIN", "OBSTRUCT"}
        assert unique_classes.issubset(expected_classes), f"Unexpected classes in dataset: {unique_classes - expected_classes}"
