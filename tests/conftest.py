"""
conftest.py - Pytest fixtures and shared configuration for IoT Drainage Pipeline E2E tests.
"""

import os
import sys
import pytest
import joblib
import pandas as pd

# Add project root to sys.path so modules (train_model, app) can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable AppTest testing framework access to plotly_chart nodes in Streamlit element tree
try:
    import streamlit.testing.v1.element_tree as _st_et
    import streamlit.testing.v1.app_test as _st_at
    if not hasattr(_st_et.ElementTree, "plotly_chart"):
        _st_et.ElementTree.plotly_chart = property(lambda self: self.get("plotly_chart"))
    if not hasattr(_st_at.AppTest, "plotly_chart"):
        _st_at.AppTest.plotly_chart = property(lambda self: self.get("plotly_chart"))
except Exception:
    pass


# Canonical file paths
GUIDE_PATH = os.path.join(PROJECT_ROOT, "AZURE_INTEGRATION_GUIDE.md")
TRAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "train_model.py")
MODEL_PATH = os.path.join(PROJECT_ROOT, "drainage_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "drainage_data.csv")
APP_PATH = os.path.join(PROJECT_ROOT, "app.py")
INO_PATH = os.path.join(PROJECT_ROOT, "FINAL_CODE_DRAINAGE_V1.ino")
TEST_INFRA_PATH = os.path.join(PROJECT_ROOT, "TEST_INFRA.md")


@pytest.fixture(scope="session")
def project_root():
    """Returns the absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def file_paths():
    """Returns a dictionary of canonical project file paths."""
    return {
        "guide": GUIDE_PATH,
        "train_script": TRAIN_SCRIPT_PATH,
        "model": MODEL_PATH,
        "data": DATA_PATH,
        "app": APP_PATH,
        "ino": INO_PATH,
        "test_infra": TEST_INFRA_PATH,
    }


@pytest.fixture(scope="session")
def loaded_model():
    """Loads and provides the serialized Random Forest model artifact."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at: {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    return model


@pytest.fixture(scope="session")
def raw_dataset():
    """Loads and returns the raw telemetry CSV dataset as a pandas DataFrame."""
    assert os.path.exists(DATA_PATH), f"Dataset file missing at: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df


@pytest.fixture
def canonical_vectors():
    """
    Returns canonical sensor test vectors with their expected classifications
    derived from physical hydraulics and firmware thresholds in FINAL_CODE_DRAINAGE_V1.ino.
    """
    return [
        {
            "description": "Normal dry-weather clear channel",
            "distance_cm": 24.2,
            "flow_l_min": 0.5,
            "expected_status": "NORMAL",
            "expected_alert_level": "LEVEL 0: NORMAL MONITORING",
            "expected_risk": "Low / Safe (Green)"
        },
        {
            "description": "Warning rising water level with moderate runoff",
            "distance_cm": 22.0,
            "flow_l_min": 1.2,
            "expected_status": "WARNING",
            "expected_alert_level": "LEVEL 1: FLOOD ADVISORY",
            "expected_risk": "Moderate / Advisory (Yellow)"
        },
        {
            "description": "Heavy storm surge downpour with high flow velocity",
            "distance_cm": 20.3,
            "flow_l_min": 5.0,
            "expected_status": "HVYRAIN",
            "expected_alert_level": "LEVEL 2: FLOOD SURGE ALERT",
            "expected_risk": "Elevated / Storm Surge (Orange)"
        },
        {
            "description": "Severe solid waste debris culvert obstruction",
            "distance_cm": 20.3,
            "flow_l_min": 1.0,
            "expected_status": "OBSTRUCT",
            "expected_alert_level": "LEVEL 3: CRITICAL ALARM (OBSTRUCTION DETECTED)",
            "expected_risk": "CRITICAL / OVERFLOW IMMINENT (Red)"
        }
    ]
