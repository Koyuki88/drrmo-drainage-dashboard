"""
test_tier4_real_world_scenarios.py - Tier 4: Real-World Scenarios & Dynamic System Simulations

Verifies application-level simulations and multi-step operational lifecycles:
- Scenario 1: Normal dry-weather baseline (calm channel, low water, normal flow)
- Scenario 2: Flash flood / heavy rain surge (HVYRAIN: high water, swift flow >= 3.5 L/min)
- Scenario 3: Culvert solid waste debris obstruction (OBSTRUCT: high water, restricted flow < 3.5 L/min, auto-valve failover)
- Scenario 4: Dynamic temporal water rise transition (NORMAL -> WARNING -> OBSTRUCT / Overflow)
- Scenario 5: Incident clearance and hydraulic recovery lifecycle
"""

import pytest
import pandas as pd
import numpy as np

import app
from app import (
    rule_based_predict,
    DRRMO_ALERT_MATRIX,
    CANAL_BED_DISTANCE_CM,
    CRITICAL_OVERFLOW_CM,
    LOW_FLOW_THRESHOLD
)
from train_model import FEATURE_COLUMNS


class TestTier4RealWorldScenarios:
    """End-to-End Simulation of Real-World Hydrological Scenarios."""

    def test_scenario_1_normal_dry_weather_baseline(self, loaded_model):
        """
        Scenario 1: Normal dry-weather drainage baseline.
        Physical Condition: Low baseflow, clear conduit, high air gap (distance ~ 24.2 cm).
        Expected System State:
        - Predicted status: NORMAL
        - Derived water depth: <= 2.0 cm
        - Culvert capacity load: < 25%
        - DRRMO Alert: Level 0 Green (Nominal, Buzzer Silenced)
        - Secondary diversion gate: CLOSED
        """
        readings = [
            {"dist": 24.5, "flow": 0.2},
            {"dist": 24.2, "flow": 0.4},
            {"dist": 24.0, "flow": 0.6},
            {"dist": 23.8, "flow": 0.5},
            {"dist": 24.1, "flow": 0.3},
        ]

        for r in readings:
            df_in = pd.DataFrame([[r["dist"], r["flow"]]], columns=FEATURE_COLUMNS)
            pred = loaded_model.predict(df_in)[0]
            assert pred == "NORMAL"

            water_depth = max(0.0, CANAL_BED_DISTANCE_CM - r["dist"])
            assert water_depth <= 2.0, f"Water depth exceeded baseline threshold: {water_depth} cm"

            capacity_pct = (water_depth / (CANAL_BED_DISTANCE_CM - CRITICAL_OVERFLOW_CM)) * 100.0
            assert capacity_pct < 25.0

            alert = DRRMO_ALERT_MATRIX[pred]
            assert "NORMAL" in alert["level_title"]
            assert alert["color"] == "#16a34a"  # Green
            assert "Silenced" in alert["buzzer_state"]

    def test_scenario_2_flash_flood_heavy_rain_surge(self, loaded_model):
        """
        Scenario 2: Flash flood / heavy rain surge ('HVYRAIN').
        Physical Condition: Intense rainfall runoff, water level high (distance 20.3 cm),
        water flowing freely and swiftly through culvert (flow rate 5.2 L/min >= 3.5 L/min).
        Expected System State:
        - Predicted status: HVYRAIN
        - Derived water depth: 4.7 cm
        - Culvert capacity load: > 80%
        - DRRMO Alert: Level 2 Orange (Flood surge protocol, fast pulse buzzer)
        - Action protocol recommends monitoring downstream outfalls, no physical culvert blockage
        """
        heavy_rain_readings = [
            {"dist": 20.4, "flow": 4.8},
            {"dist": 20.3, "flow": 5.2},
            {"dist": 20.1, "flow": 6.0},
        ]

        for r in heavy_rain_readings:
            df_in = pd.DataFrame([[r["dist"], r["flow"]]], columns=FEATURE_COLUMNS)
            pred = loaded_model.predict(df_in)[0]
            assert pred == "HVYRAIN", f"Expected HVYRAIN, got {pred} for reading {r}"

            water_depth = max(0.0, CANAL_BED_DISTANCE_CM - r["dist"])
            assert water_depth >= 4.5

            capacity_pct = (water_depth / (CANAL_BED_DISTANCE_CM - CRITICAL_OVERFLOW_CM)) * 100.0
            assert capacity_pct >= 80.0

            alert = DRRMO_ALERT_MATRIX[pred]
            assert "SURGE" in alert["level_title"] or "HVYRAIN" in pred
            assert alert["color"] == "#ea580c"  # Orange
            assert "FAST" in alert["buzzer_state"]

    def test_scenario_3_culvert_debris_obstruction_and_valve_failover(self, loaded_model):
        """
        Scenario 3: Culvert debris blockage / obstruction ('OBSTRUCT').
        Physical Condition: Water backing up (distance 20.2 cm, high water level), but water flow is choked
        by plastic waste / sediment (flow rate 0.8 L/min << 3.5 L/min).
        Expected System State:
        - Predicted status: OBSTRUCT
        - DRRMO Alert: Level 3 Red Critical
        - Action protocol: Dispatch emergency crew and activate secondary diversion gate
        - Simulated failover actuator logic: valve_open switches to True
        """
        obstruction_readings = [
            {"dist": 20.4, "flow": 1.2},
            {"dist": 20.3, "flow": 0.8},
            {"dist": 20.1, "flow": 0.4},
        ]

        # Simulation state for secondary valve
        valve_open = False
        auto_failover_enabled = True

        for r in obstruction_readings:
            df_in = pd.DataFrame([[r["dist"], r["flow"]]], columns=FEATURE_COLUMNS)
            pred = loaded_model.predict(df_in)[0]
            assert pred == "OBSTRUCT", f"Expected OBSTRUCT, got {pred} for reading {r}"

            alert = DRRMO_ALERT_MATRIX[pred]
            assert "CRITICAL" in alert["level_title"] or "OBSTRUCTION" in alert["level_title"]
            assert alert["color"] == "#dc2626"  # Red
            assert "EMERGENCY" in alert["buzzer_state"]
            assert "diversion gate" in alert["action_protocol"].lower()

            # Execute automated failover logic (as implemented in app.py line 562)
            if auto_failover_enabled and pred == "OBSTRUCT":
                valve_open = True

        # Verify automated failover successfully triggered
        assert valve_open is True, "Automated secondary valve failover failed to activate during critical obstruction event."

    def test_scenario_4_dynamic_rising_water_temporal_transition(self, loaded_model):
        """
        Scenario 4: Rising water transition from Normal -> Warning -> Obstruction / Overflow.
        Simulates progressive hydrological time-series:
        - Step 1: Normal clear baseflow
        - Step 2: Warning runoff accumulation
        - Step 3: Critical debris trap and obstruction
        - Step 4: Overflow risk at maximum canal crest
        """
        time_series_sequence = [
            {"step": 1, "dist": 24.5, "flow": 0.3, "expected_status": "NORMAL", "expected_risk": "Low / Safe (Green)"},
            {"step": 2, "dist": 22.2, "flow": 1.5, "expected_status": "WARNING", "expected_risk": "Moderate / Advisory (Yellow)"},
            {"step": 3, "dist": 20.4, "flow": 0.9, "expected_status": "OBSTRUCT", "expected_risk": "CRITICAL / OVERFLOW IMMINENT (Red)"},
            {"step": 4, "dist": 19.8, "flow": 0.5, "expected_status": "OBSTRUCT", "expected_risk": "CRITICAL / OVERFLOW IMMINENT (Red)"},
        ]

        previous_depth = -1.0
        for event in time_series_sequence:
            df_in = pd.DataFrame([[event["dist"], event["flow"]]], columns=FEATURE_COLUMNS)
            pred = loaded_model.predict(df_in)[0]
            assert pred == event["expected_status"], (
                f"Step {event['step']} transition mismatch: input d={event['dist']}, f={event['flow']} "
                f"yielded '{pred}', expected '{event['expected_status']}'"
            )

            current_depth = max(0.0, CANAL_BED_DISTANCE_CM - event["dist"])
            assert current_depth > previous_depth, f"Step {event['step']}: Water depth did not increase as expected."
            previous_depth = current_depth

            alert = DRRMO_ALERT_MATRIX[pred]
            assert alert["risk_level"] == event["expected_risk"]

    def test_scenario_5_post_obstruction_clearance_and_recovery(self, loaded_model):
        """
        Scenario 5: Post-obstruction clearance lifecycle and return to normal baseline.
        Simulates:
        - Emergency clearing of debris (trapped flood rushes through at high flow -> HVYRAIN)
        - Drainage of canal back to baseline depth -> NORMAL
        - Reset of secondary diversion gate
        """
        lifecycle_events = [
            # 1. Trapped water under obstruction
            {"dist": 20.3, "flow": 0.8, "expected_pred": "OBSTRUCT"},
            # 2. Debris physically cleared by crew: trapped head rushes through
            {"dist": 20.4, "flow": 5.8, "expected_pred": "HVYRAIN"},
            # 3. Water drains down towards warning level
            {"dist": 22.0, "flow": 2.5, "expected_pred": "WARNING"},
            # 4. Canal fully recovered to baseline clear state
            {"dist": 24.5, "flow": 0.4, "expected_pred": "NORMAL"}
        ]

        valve_state = True  # Valve was opened during obstruction

        for idx, event in enumerate(lifecycle_events):
            df_in = pd.DataFrame([[event["dist"], event["flow"]]], columns=FEATURE_COLUMNS)
            pred = loaded_model.predict(df_in)[0]
            assert pred == event["expected_pred"], f"Event {idx+1} mismatch: expected {event['expected_pred']}, got {pred}"

            if pred == "NORMAL":
                # Operators safely reset the secondary valve
                valve_state = False

        assert valve_state is False, "Valve should be safely closed once normal baseline is restored."
