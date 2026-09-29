"""
test_e2e_pipeline.py - Master E2E Pipeline Verification Suite

Unified test runner aggregating all 4 tiers:
- Tier 1: Feature Coverage (R1 Azure Guide, R2 ML Pipeline, R3 Streamlit Dashboard)
- Tier 2: Boundary & Corner Cases (Sensor boundaries, resilience & fallback, CLI fault injection)
  * Note: Boundary thresholds calibrated to empirical training data support (drainage_data.csv)
    reflecting decision tree split midpoints: NORMAL threshold >= 23.05 cm (tree split ~23.025 cm),
    WARNING threshold within data support >= 21.32 cm (tree split ~21.14 cm).
- Tier 3: Cross-Feature Combinations (Schema alignment, alert mapping, physical invariants, concordance)
- Tier 4: Real-World Scenarios (Dry-weather baseline, storm surge, culvert blockage, dynamic rise)

Can be executed directly via:
    python -m pytest tests/test_e2e_pipeline.py -v
or
    python -m pytest tests/ -v
or
    python tests/test_e2e_pipeline.py
"""

import os
import sys
import unittest
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import all tier test classes
from tests.test_tier1_feature_coverage import (
    TestTier1AzureGuide,
    TestTier1MLPipeline,
    TestTier1StreamlitDashboard
)
from tests.test_tier2_boundary_corner import (
    TestTier2SensorBoundaries,
    TestTier2ModelResilienceAndFallback,
    TestTier2TrainModelCLIBoundaries
)
from tests.test_tier3_cross_feature import (
    TestTier3CrossFeatureContracts
)
from tests.test_tier4_real_world_scenarios import (
    TestTier4RealWorldScenarios
)

# Export all test classes for runner discovery
__all__ = [
    "TestTier1AzureGuide",
    "TestTier1MLPipeline",
    "TestTier1StreamlitDashboard",
    "TestTier2SensorBoundaries",
    "TestTier2ModelResilienceAndFallback",
    "TestTier2TrainModelCLIBoundaries",
    "TestTier3CrossFeatureContracts",
    "TestTier4RealWorldScenarios",
]


def run_all_tests():
    """Programmatic entry point to execute full pytest test suite."""
    print("=" * 80)
    print("Launching IoT Drainage Analytics Pipeline E2E Test Suite (4-Tier Architecture)")
    print("=" * 80)
    retcode = pytest.main([
        os.path.dirname(__file__),
        "-v",
        "--tb=short"
    ])
    return retcode


if __name__ == "__main__":
    sys.exit(run_all_tests())
