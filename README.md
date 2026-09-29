# DRRMO Smart Drainage & Flood Early Warning System

> **IoT-Enabled Drainage Obstruction Detection System with Water Overflow Risk Prediction and Automated Alert Mechanism**

A real-time drainage monitoring dashboard built for the **Imus City CDRRMO**, Cavite. The system uses an ESP32 microcontroller with dual sensors (ultrasonic + water flow) to classify drainage states and predict overflow risk using a Random Forest Machine Learning model.

## Features

- **4-State Drainage Classification:** NORMAL · WARNING · HVYRAIN · OBSTRUCT
- **Random Forest ML Model:** Trained on 1,039 prototype telemetry records
- **Enterprise UI:** Industrial-grade Streamlit dashboard with zero rounded corners, crimson accent palette, and tactical loading indicators
- **Real-Time KPI Telemetry:** Water depth, flow rate, ML confidence score, and culvert capacity utilization
- **Interactive Plotly Charts:** Dual-Y axis time-series with inverted distance axis
- **DRRMO Alert Matrix:** Color-coded 4-tier alert framework with recommended action protocols
- **Azure Cloud Architecture:** Complete integration guide for IoT Hub → Functions → Cosmos DB

## Tech Stack

| Layer | Technology |
|---|---|
| Hardware | ESP32 DevKit, JSN-SR04T Ultrasonic, YF-S201 / YF-DN50 Flow Sensor |
| ML Model | scikit-learn RandomForestClassifier |
| Dashboard | Streamlit 1.64, Plotly 6.5 |
| Cloud (Planned) | Microsoft Azure IoT Hub, Azure Functions v2, Cosmos DB |

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/drrmo-drainage-dashboard.git
cd drrmo-drainage-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
drrmo-drainage-dashboard/
├── .streamlit/
│   └── config.toml              # Enterprise theme (locked to light mode)
├── tests/
│   ├── conftest.py              # Shared test fixtures
│   ├── test_e2e_pipeline.py     # End-to-end integration tests
│   ├── test_tier1_feature_coverage.py
│   ├── test_tier2_boundary_corner.py
│   ├── test_tier3_cross_feature.py
│   └── test_tier4_real_world_scenarios.py
├── app.py                       # Streamlit Command Center Dashboard (949 lines)
├── train_model.py               # ML training pipeline
├── drainage_model.pkl           # Serialized Random Forest model
├── drainage_data.csv            # Prototype telemetry dataset (1,039 records)
├── eda_visualizations.py        # Exploratory Data Analysis script
├── AZURE_INTEGRATION_GUIDE.md   # Cloud architecture blueprint
├── requirements.txt
└── README.md
```

## ML Model Performance

| Metric | Score |
|---|---|
| Accuracy | 1.0000 |
| Precision (Macro) | 1.0000 |
| Recall (Macro) | 1.0000 |

> Model trained on controlled PVC prototype data. Will be recalibrated with live field data from Imus City deployment.

## Testing

```bash
# Run the full 88-test verification suite
python -m pytest tests/ -v
```

## Authors

- Asas, Hanna Jane N.
- Corrales, Prince Joyous E.
- Cuenca, Zeruel Kody C.
- Opeña, Jann Rapha-el S.
- Sobrevega, Kenjie T.

**Department of Computer, Electronics, and Electrical Engineering**
College of Engineering and Information Technology — Cavite State University

## License

This project is developed as an academic capstone. All rights reserved.
