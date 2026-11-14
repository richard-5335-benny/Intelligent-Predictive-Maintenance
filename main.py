import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate simulated industrial motor sensor data

np.random.seed(42)

# Number of sensor readings
num_readings = 100

# Time intervals in seconds
time = np.arange(0, num_readings * 5, 5)

# Simulated temperature readings (°C)
temperature = np.random.normal(55, 5, num_readings)

# Simulated vibration readings (mm/s)
vibration = np.random.normal(3.5, 0.8, num_readings)

# Simulated motor speed (RPM)
rpm = np.random.normal(1500, 50, num_readings)

# Simulated operating hours
operating_hours = np.linspace(100, 200, num_readings)

# Create a structured sensor dataset

data = pd.DataFrame({
    "Time (seconds)": time,
    "Temperature (C)": temperature,
    "Vibration (mm/s)": vibration,
    "Motor Speed (RPM)": rpm,
    "Operating Hours": operating_hours
})

# Display the first 10 readings
print(data.head(10))

# Save sensor data

data.to_csv("data/sensor_data.csv", index=False)

print("\nSensor data generated successfully!")
print("Dataset saved in data/sensor_data.csv")

# Stage 2: Sensor Data Visualisation

# Temperature graph
plt.figure(figsize=(10, 5))

plt.plot(
    data["Time (seconds)"],
    data["Temperature (C)"],
    label="Motor Temperature"
)

plt.title("Industrial Motor Temperature Monitoring")
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (C)")
plt.legend()
plt.grid(True)

plt.savefig("graphs/temperature_graph.png", dpi=300)

# Vibration graph
plt.figure(figsize=(10, 5))

plt.plot(
    data["Time (seconds)"],
    data["Vibration (mm/s)"],
    label="Motor Vibration"
)

plt.title("Industrial Motor Vibration Monitoring")
plt.xlabel("Time (seconds)")
plt.ylabel("Vibration (mm/s)")
plt.legend()
plt.grid(True)

plt.savefig("graphs/vibration_graph.png", dpi=300)

# Motor speed graph
plt.figure(figsize=(10, 5))

plt.plot(
    data["Time (seconds)"],
    data["Motor Speed (RPM)"],
    label="Motor Speed"
)

plt.title("Industrial Motor Speed Monitoring")
plt.xlabel("Time (seconds)")
plt.ylabel("Motor Speed (RPM)")
plt.legend()
plt.grid(True)

plt.savefig("graphs/motor_speed_graph.png", dpi=300)

# Display all three graphs
plt.show()

print("\nAll sensor graphs generated successfully!")

# Stage 3: Threshold-Based Fault Detection

# Define sensor thresholds
TEMP_LIMIT = 60
VIBRATION_LIMIT = 5
RPM_MIN = 1400
RPM_MAX = 1600

# Temperature fault detection
data["Temperature Status"] = np.where(
    data["Temperature (C)"] > TEMP_LIMIT,
    "HIGH TEMPERATURE",
    "NORMAL"
)

# Vibration fault detection
data["Vibration Status"] = np.where(
    data["Vibration (mm/s)"] > VIBRATION_LIMIT,
    "HIGH VIBRATION",
    "NORMAL"
)

# Motor speed fault detection
data["Speed Status"] = np.where(
    (data["Motor Speed (RPM)"] < RPM_MIN) |
    (data["Motor Speed (RPM)"] > RPM_MAX),
    "ABNORMAL SPEED",
    "NORMAL"
)

# Display abnormal readings
print("\n--- MOTOR FAULT DETECTION REPORT ---")

abnormal_data = data[
    (data["Temperature Status"] != "NORMAL") |
    (data["Vibration Status"] != "NORMAL") |
    (data["Speed Status"] != "NORMAL")
]

print(abnormal_data.to_string(index=False))

# Save fault detection report
abnormal_data.to_csv(
    "reports/fault_detection_report.csv",
    index=False
)

print("\nFault detection completed successfully!")


# ==========================================
# STAGE 4: STATISTICAL ANALYSIS
# ==========================================

# Select numerical sensor columns
sensor_columns = [
    "Temperature (C)",
    "Vibration (mm/s)",
    "Motor Speed (RPM)",
    "Operating Hours"
]

# Calculate statistical measures
statistics = data[sensor_columns].agg(
    ["mean", "median", "std", "min", "max"]
)

# Display statistical summary
print("\n--- SENSOR STATISTICAL ANALYSIS ---")
print(statistics)

# Save statistical report
statistics.to_csv("reports/statistical_summary.csv")

print("\nStatistical analysis completed successfully!")

# ==========================================
# STAGE 5: MACHINE LEARNING
# ISOLATION FOREST ANOMALY DETECTION
# ==========================================

features = data[
    [
        "Temperature (C)",
        "Vibration (mm/s)",
        "Motor Speed (RPM)",
        "Operating Hours"
    ]
]

print("\n--- MACHINE LEARNING INPUT FEATURES ---")
print(features.head())

# Create Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

print("\nIsolation Forest model created successfully!")

# Train the Isolation Forest model
data["ML Prediction"] = model.fit_predict(features)

print("\nModel training completed successfully!")

print("\n--- MACHINE LEARNING PREDICTIONS ---")
print(data[["Temperature (C)", "Vibration (mm/s)",
            "Motor Speed (RPM)", "ML Prediction"]].head(10))

# Convert ML predictions into readable labels

data["ML Anomaly Status"] = data["ML Prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

print("\n--- ANOMALY STATUS ---")
print(data[["ML Prediction", "ML Anomaly Status"]].head(10))

# Extract anomalous readings

anomalies = data[
    data["ML Prediction"] == -1
].copy()

print("\n--- ISOLATION FOREST ANOMALY REPORT ---")

print("Total Sensor Readings:", len(data))
print("Detected Anomalies:", len(anomalies))

print("\nAnomalous Sensor Readings:")

print(anomalies.to_string(index=False))

# Save anomaly report

anomalies.to_csv(
    "reports/ml_anomaly_report.csv",
    index=False
)

print("\nMachine Learning Analysis Completed!")
print("Anomaly report saved successfully!")

