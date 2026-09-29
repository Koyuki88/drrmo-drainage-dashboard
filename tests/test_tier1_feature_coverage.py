"""
test_tier1_feature_coverage.py - Tier 1: Feature Coverage E2E Tests

Verifies primary functional requirements (>= 5 test cases per feature for R1, R2, and R3):
- R1: Azure Database Integration Guide (File, CLI commands, Diagrams, Python v2, Schema/TTL)
- R2: Machine Learning Pipeline (Execution, stdout metrics, pickle artifact, interface, canonical inference)
- R3: Streamlit Dashboard (Syntax, model loading, AppTest headless run, DRRMO alerts, KPI metrics, chart, ML explainability)
"""

import os
import sys
import ast
import re
import subprocess
import tempfile
import pytest
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


class TestTier1AzureGuide:
    """R1: Azure Database Integration Guide Verification (>= 5 test cases)."""

    def test_r1_guide_file_exists_and_substantial(self, file_paths):
        """Case 1: Verify AZURE_INTEGRATION_GUIDE.md exists and is substantive (> 20KB)."""
        guide_path = file_paths["guide"]
        assert os.path.exists(guide_path), f"AZURE_INTEGRATION_GUIDE.md not found at {guide_path}"
        file_size = os.path.getsize(guide_path)
        assert file_size > 20_000, f"Guide file size ({file_size} bytes) is suspiciously small; expected > 20KB."

    def test_r1_guide_contains_specific_azure_cli_commands(self, file_paths):
        """Case 2: Verify specific required Azure CLI provisioning commands are present."""
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            content = f.read()

        required_cli_patterns = [
            r"az\s+iot\s+hub\s+create",
            r"az\s+cosmosdb\s+create",
            r"az\s+cosmosdb\s+sql\s+container\s+create",
            r"az\s+functionapp\s+create",
            r"az\s+cosmosdb\s+sql\s+role\s+assignment\s+create",
        ]
        for pattern in required_cli_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            assert match is not None, f"Missing required Azure CLI command matching pattern: '{pattern}' in guide."

    def test_r1_guide_contains_ascii_and_mermaid_diagrams(self, file_paths):
        """Case 3: Verify presence of both ASCII architecture diagram and Mermaid flow/sequence diagrams."""
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            content = f.read()

        # Check for ASCII architecture diagram
        assert "END-TO-END SYSTEM ARCHITECTURE" in content or "[ ESP32 Edge Node ]" in content, \
            "ASCII architecture diagram missing from guide."

        # Check for Mermaid diagrams
        assert "```mermaid" in content, "Mermaid diagram block missing from guide."
        mermaid_types = ["flowchart", "sequenceDiagram"]
        for m_type in mermaid_types:
            assert m_type in content, f"Mermaid '{m_type}' diagram type missing from guide."

    def test_r1_guide_contains_valid_python_v2_azure_function(self, file_paths):
        """Case 4: Extract and validate syntax of Python v2 Azure Function code blocks."""
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            content = f.read()

        # Verify presence of Python v2 programming model decorators
        assert "@app.event_hub_message_trigger" in content, "Missing '@app.event_hub_message_trigger' binding."
        assert "@app.cosmos_db_output" in content, "Missing '@app.cosmos_db_output' binding."

        # Extract all Python code blocks from markdown
        python_blocks = re.findall(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
        assert len(python_blocks) > 0, "No Python code blocks found in AZURE_INTEGRATION_GUIDE.md."

        # Find the function_app block and verify it parses cleanly via AST
        func_app_block = None
        for block in python_blocks:
            if "process_drainage_telemetry_batch" in block or "event_hub_message_trigger" in block:
                func_app_block = block
                break

        assert func_app_block is not None, "Azure Function implementation block not found in guide."
        # Validate syntax via abstract syntax tree parsing
        parsed_ast = ast.parse(func_app_block)
        assert isinstance(parsed_ast, ast.Module), "Failed to parse Azure Function code block as valid Python AST."

    def test_r1_guide_defines_partition_key_and_ttl(self, file_paths):
        """Case 5: Verify guide defines Cosmos DB partition key '/deviceId' and TTL policy."""
        with open(file_paths["guide"], "r", encoding="utf-8") as f:
            content = f.read()

        # Partition key /deviceId
        assert "/deviceId" in content, "Cosmos DB partition key '/deviceId' missing from guide."
        assert "--partition-key-path" in content, "CLI parameter '--partition-key-path' missing from guide."

        # TTL definition (90 days / default-ttl)
        assert "ttl" in content.lower(), "TTL documentation missing from guide."
        assert "--default-ttl" in content, "CLI parameter '--default-ttl' missing from guide."


class TestTier1MLPipeline:
    """R2: Machine Learning Pipeline Verification (>= 5 test cases)."""

    def test_r2_train_model_executes_exit_code_0(self, file_paths):
        """Case 1: Execute train_model.py via subprocess and verify clean exit code 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model = os.path.join(tmp_dir, "test_model.pkl")
            result = subprocess.run(
                [sys.executable, file_paths["train_script"], "--output", tmp_model, "--estimators", "20"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(file_paths["train_script"])
            )
            assert result.returncode == 0, f"train_model.py failed with exit code {result.returncode}.\nStderr: {result.stderr}"
            assert os.path.exists(tmp_model), f"Output model was not created at: {tmp_model}"

    def test_r2_stdout_reports_accuracy_precision_recall(self, file_paths):
        """Case 2: Verify train_model.py stdout prints accuracy, precision (macro & weighted), and recall (macro & weighted)."""
        result = subprocess.run(
            [sys.executable, file_paths["train_script"], "--no-verify", "--estimators", "20"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_paths["train_script"])
        )
        assert result.returncode == 0
        stdout = result.stdout

        required_metrics = [
            "Accuracy:",
            "Precision (Macro):",
            "Precision (Weighted):",
            "Recall (Macro):",
            "Recall (Weighted):",
        ]
        for metric in required_metrics:
            assert metric in stdout, f"Expected metric '{metric}' not printed to stdout.\nStdout was:\n{stdout}"

        # Verify accuracy is a valid float >= 0.95
        acc_match = re.search(r"Accuracy:\s+([0-9\.]+)", stdout)
        assert acc_match is not None, "Could not extract numeric accuracy from stdout."
        acc_value = float(acc_match.group(1))
        assert acc_value >= 0.95, f"Expected accuracy >= 0.95, got {acc_value}"

    def test_r2_model_pickle_artifact_generated_and_valid_size(self, file_paths):
        """Case 3: Verify drainage_model.pkl exists and has valid serialization size (> 50KB)."""
        model_path = file_paths["model"]
        assert os.path.exists(model_path), f"drainage_model.pkl missing at: {model_path}"
        file_size = os.path.getsize(model_path)
        assert file_size > 50_000, f"Model file size ({file_size} bytes) is suspiciously small; expected > 50KB."

    def test_r2_model_loadable_and_has_required_interface(self, loaded_model):
        """Case 4: Verify loaded model exposes predict, predict_proba, classes_, and all 4 target classes."""
        assert hasattr(loaded_model, "predict"), "Model artifact missing 'predict' method."
        assert hasattr(loaded_model, "predict_proba"), "Model artifact missing 'predict_proba' method."
        assert hasattr(loaded_model, "classes_"), "Model artifact missing 'classes_' attribute."

        expected_classes = {"HVYRAIN", "NORMAL", "OBSTRUCT", "WARNING"}
        actual_classes = set(loaded_model.classes_)
        assert actual_classes == expected_classes, f"Class mismatch. Expected {expected_classes}, got {actual_classes}"

    def test_r2_model_predicts_known_vectors_accurately(self, loaded_model, canonical_vectors):
        """Case 5: Verify model yields 100% accurate predictions for all canonical operational vectors."""
        feature_names = ["distance_cm", "flow_l_min"]
        for vector in canonical_vectors:
            sample_df = pd.DataFrame(
                [[vector["distance_cm"], vector["flow_l_min"]]],
                columns=feature_names
            )
            prediction = loaded_model.predict(sample_df)[0]
            assert prediction == vector["expected_status"], (
                f"Failed prediction for {vector['description']}: "
                f"Input [d={vector['distance_cm']}, f={vector['flow_l_min']}] -> "
                f"Got '{prediction}', Expected '{vector['expected_status']}'"
            )
            probabilities = loaded_model.predict_proba(sample_df)[0]
            pred_idx = list(loaded_model.classes_).index(prediction)
            # Confidence score for ground truth should be significant
            assert probabilities[pred_idx] > 0.50, f"Confidence too low for {prediction}: {probabilities[pred_idx]}"


class TestTier1StreamlitDashboard:
    """R3: Streamlit Dashboard Verification (>= 5 test cases)."""

    def test_r3_app_script_exists_and_compiles(self, file_paths):
        """Case 1: Verify app.py exists and compiles cleanly without syntax errors."""
        app_path = file_paths["app"]
        assert os.path.exists(app_path), f"app.py not found at: {app_path}"

        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()

        # AST compile check
        parsed = ast.parse(source, filename=app_path)
        assert isinstance(parsed, ast.Module), "Failed to compile app.py into Python AST."
        # Bytecode compilation check
        code_obj = compile(source, app_path, "exec")
        assert code_obj is not None, "Bytecode compilation failed for app.py."

    def test_r3_app_loads_model_with_cache_resource(self, file_paths):
        """Case 2: Verify app.py defines load_ml_model with @st.cache_resource."""
        with open(file_paths["app"], "r", encoding="utf-8") as f:
            content = f.read()

        assert "load_ml_model" in content, "Function 'load_ml_model' missing from app.py."
        assert "@st.cache_resource" in content, "Decorator '@st.cache_resource' missing from app.py."

    def test_r3_app_renders_via_apptest_without_exceptions(self, file_paths):
        """Case 3: Execute app.py headlessly using streamlit.testing.v1.AppTest and check for zero unhandled exceptions."""
        at = AppTest.from_file(file_paths["app"], default_timeout=30)
        at.run()
        assert len(at.exception) == 0, f"AppTest execution raised exceptions: {[e.message for e in at.exception]}"

    def test_r3_app_renders_drrmo_alert_components(self, file_paths):
        """Case 4: Verify DRRMO alert components and severity banners render in dashboard."""
        at = AppTest.from_file(file_paths["app"], default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # Check rendered markdown contains DRRMO alert identifiers
        rendered_markdown_texts = [m.value for m in at.markdown]
        full_text = " ".join(rendered_markdown_texts)
        assert "DRRMO" in full_text, "DRRMO branding/alert missing from rendered markdown."
        assert any(term in full_text for term in ["NORMAL", "WARNING", "HVYRAIN", "OBSTRUCT"]), \
            "Rendered dashboard does not contain any DRRMO alert status indicators."

    def test_r3_app_renders_kpi_metrics(self, file_paths):
        """Case 5: Verify st.metric KPI cards render (Water Depth, Flow Rate, Status, Confidence)."""
        at = AppTest.from_file(file_paths["app"], default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        assert len(at.metric) >= 4, f"Expected at least 4 KPI metrics, found {len(at.metric)}."
        labels = [m.label for m in at.metric]
        assert any("Water" in l or "Depth" in l for l in labels), "Water depth metric missing."
        assert any("Flow" in l for l in labels), "Flow rate metric missing."
        assert any("Status" in l for l in labels), "Drainage status metric missing."

    def test_r3_app_renders_timeseries_chart(self, file_paths):
        """Case 6: Verify dual-axis inverted time-series Plotly chart is rendered."""
        at = AppTest.from_file(file_paths["app"], default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # Verify plotly chart component exists
        assert len(at.plotly_chart) >= 1, "Time-series Plotly chart missing from rendered Streamlit DOM."

    def test_r3_app_renders_ml_prediction_outputs(self, file_paths):
        """Case 7: Verify ML prediction outputs, class probabilities, and explainability display."""
        at = AppTest.from_file(file_paths["app"], default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # In app.py, there is a second Plotly chart for probability distribution or explainability
        assert len(at.plotly_chart) >= 2, "ML probability distribution chart missing from rendered Streamlit DOM."
        # Verify confidence metric or prediction text in output
        labels = [m.label for m in at.metric]
        assert any("Confidence" in l or "Probability" in l for l in labels), "Confidence metric missing from KPIs."
