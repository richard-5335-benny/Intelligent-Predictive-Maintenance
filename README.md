
# Intelligent Predictive Maintenance System

An intelligent motor monitoring and anomaly detection application developed using Python, Machine Learning, Pandas, Matplotlib, and Streamlit.

The system analyses simulated motor sensor readings, identifies abnormal operating conditions, and presents motor performance information through an interactive dashboard.

## Project Overview

Predictive maintenance helps identify unusual equipment behaviour before it develops into a major operational problem.

This project demonstrates a basic predictive maintenance workflow using simulated motor sensor data and machine learning-based anomaly detection.

The system monitors:
- Motor temperature
- Motor vibration
- Motor speed
- Operating hours

It uses threshold-based fault detection and the Isolation Forest algorithm to identify abnormal sensor readings.

## Key Features

- Simulated motor sensor data generation
- Sensor data visualisation
- Threshold-based fault detection
- Fault severity classification
- Machine learning-based anomaly detection
- Isolation Forest algorithm
- Statistical analysis of sensor readings
- Interactive Streamlit dashboard
- CSV report generation and downloads
- Sidebar navigation across six dashboard sections

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations and data simulation |
| Pandas | Data processing and analysis |
| Matplotlib | Sensor data visualisation |
| Scikit-learn | Machine learning and anomaly detection |
| Isolation Forest | Unsupervised anomaly detection |
| Streamlit | Interactive dashboard |

## Project Structure

```text
pred_mtc/
│
├── app.py
├── app_backup.py
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sensor_data.csv
│
├── graphs/
│   ├── temperature_graph.png
│   ├── vibration_graph.png
│   └── motor_speed_graph.png
│
├── models/
│
└── reports/
    ├── fault_detection_report.csv
    ├── ml_anomaly_report.csv
    └── statistical_summary.csv
```

## System Modules

### 1. Sensor Data Generation

Generates simulated motor sensor readings using NumPy and Pandas.

The dataset contains 100 readings recorded at five-second intervals.

### 2. Sensor Data Visualisation

Uses Matplotlib to generate graphs for:
- Temperature monitoring
- Vibration monitoring
- Motor speed monitoring

### 3. Threshold-Based Fault Detection

Identifies abnormal motor conditions using predefined thresholds.

| Parameter | Threshold |
|---|---|
| Temperature | Above 60 °C |
| Vibration | Above 5 mm/s |
| Motor Speed | Below 1400 RPM or above 1600 RPM |

These are illustrative project thresholds, not manufacturer-validated operating limits.

### 4. Fault Severity Classification

Classifies sensor readings into three severity levels:

- NORMAL: No threshold violations
- WARNING: One threshold violation
- CRITICAL: Two or more threshold violations

### 5. Machine Learning-Based Anomaly Detection

Uses the Isolation Forest algorithm from Scikit-learn.

The model identifies statistically unusual sensor readings based on:

- Temperature
- Vibration
- Motor speed
- Operating hours

The model is configured with 100 estimators and a contamination parameter of 0.05.

Five anomalous readings were identified in the current simulated dataset.

### 6. Interactive Dashboard

Developed using Streamlit.

The dashboard includes six navigation sections:

1. Overview
2. Sensor Data
3. Fault Detection
4. ML Anomalies
5. Statistical Analysis
6. Download Reports

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd pred_mtc
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / Ubuntu / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Generate sensor data and reports

```bash
python main.py
```

### Step 2: Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

## Sample Results

The current simulated dataset contains:

| Metric | Result |
|---|---:|
| Total Sensor Readings | 100 |
| Average Temperature | 54.48 °C |
| Average Vibration | 3.52 mm/s |
| Average Motor Speed | 1503 RPM |
| ML Anomalies | 5 |
| Normal Readings | 80 |
| Warning Readings | 17 |
| Critical Readings | 3 |

## Limitations

- Sensor readings are simulated and are not collected from a physical motor.
- Threshold values are illustrative.
- Isolation Forest identifies statistical anomalies; it does not directly predict equipment failure.
- The current implementation does not calculate Remaining Useful Life (RUL).
- The machine learning model has not been validated using independent industrial datasets.

## Future Improvements

- Integration with real-time motor sensors.
- Arduino or ESP32-based IoT data acquisition.
- Live sensor monitoring through a cloud database.
- Comparison of Isolation Forest, Decision Tree, and Random Forest approaches.
- Model evaluation using separate training and testing datasets.
- Remaining Useful Life estimation.
- Predictive failure analysis.
- Real-time alerts and notifications.

## Author

Mechatronics Engineering Student

## License

This project is available for educational and learning purposes.