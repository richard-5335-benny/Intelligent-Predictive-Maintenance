
import streamlit as st
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================
# 2. PROFESSIONAL DASHBOARD STYLING
# ==========================================

st.markdown("""
<style>

/* Main dashboard background and spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
h1 {
    font-weight: 700;
}

/* Section headings */
h2, h3 {
    font-weight: 600;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #F0F4F8;
}

/* KPI metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    padding: 18px;
    border-radius: 12px;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-weight: 600;
}

/* Dataframe borders */
[data-testid="stDataFrame"] {
    border: 1px solid #D9E2EC;
    border-radius: 10px;
}

/* Download buttons */
.stDownloadButton button {
    border-radius: 8px;
    font-weight: 600;
}

/* Horizontal dividers */
hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Fault severity cards */
.severity-card {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}

/* Normal severity */
.severity-normal {
    background-color: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
}

/* Warning severity */
.severity-warning {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}

/* Critical severity */
.severity-critical {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}

/* Severity card headings */
.severity-card h3 {
    font-size: 16px;
    margin: 0;
}

/* Severity card values */
.severity-card p {
    font-size: 28px;
    font-weight: bold;
    margin: 8px 0 0 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD PROJECT DATA
# ==========================================

data = pd.read_csv("data/sensor_data.csv")

fault_data = pd.read_csv(
    "reports/fault_detection_report.csv"
)

anomaly_data = pd.read_csv(
    "reports/ml_anomaly_report.csv"
)

statistics = pd.read_csv(
    "reports/statistical_summary.csv",
    index_col=0
)


# Original sensor parameters
sensor_parameters = [
    "Time (seconds)",
    "Temperature (C)",
    "Vibration (mm/s)",
    "Motor Speed (RPM)",
    "Operating Hours"
]


# ==========================================
# 4. FAULT SEVERITY CLASSIFICATION
# ==========================================

data["High Temperature"] = (
    data["Temperature (C)"] > 60
)

data["High Vibration"] = (
    data["Vibration (mm/s)"] > 5
)

data["Abnormal Speed"] = (
    (data["Motor Speed (RPM)"] < 1400) |
    (data["Motor Speed (RPM)"] > 1600)
)

data["Fault Count"] = (
    data["High Temperature"].astype(int) +
    data["High Vibration"].astype(int) +
    data["Abnormal Speed"].astype(int)
)

data["Fault Severity"] = data["Fault Count"].apply(
    lambda count:
        "CRITICAL" if count >= 2
        else "WARNING" if count == 1
        else "NORMAL"
)


# ==========================================
# 5. FAULT SEVERITY DISPLAY FUNCTION
# ==========================================

def display_severity_card(label, count, severity):

    st.markdown(
        f"""
        <div class="severity-card severity-{severity.lower()}">
            <h3>{label}</h3>
            <p>{count}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 6. DASHBOARD HEADER
# ==========================================

st.title("Intelligent Predictive Maintenance System")

st.write(
    "Monitor motor sensor readings and detect abnormal operating conditions."
)

st.divider()


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Select a Section",
    [
        "Overview",
        "Sensor Data",
        "Fault Detection",
        "ML Anomalies",
        "Statistical Analysis",
        "Download Reports"
    ]
)


# ==========================================
# PAGE 1 - OVERVIEW
# ==========================================

if page == "Overview":

    st.header("Motor Performance Overview")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Temperature",
        f"{data['Temperature (C)'].mean():.2f} °C"
    )

    col2.metric(
        "Average Vibration",
        f"{data['Vibration (mm/s)'].mean():.2f} mm/s"
    )

    col3.metric(
        "Average Motor Speed",
        f"{data['Motor Speed (RPM)'].mean():.0f} RPM"
    )

    col4.metric(
        "ML Anomalies",
        len(anomaly_data)
    )

    st.divider()

    # Dataset information
    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    info_col1.metric(
        "Total Sensor Readings",
        len(data)
    )

    info_col2.metric(
        "Number of Parameters",
        len(sensor_parameters)
    )

    st.divider()

    # Fault severity summary
    st.subheader("Fault Severity Summary")

    normal_count = (
        data["Fault Severity"] == "NORMAL"
    ).sum()

    warning_count = (
        data["Fault Severity"] == "WARNING"
    ).sum()

    critical_count = (
        data["Fault Severity"] == "CRITICAL"
    ).sum()

    severity_col1, severity_col2, severity_col3 = st.columns(3)

    with severity_col1:
        display_severity_card(
            "NORMAL",
            int(normal_count),
            "NORMAL"
        )

    with severity_col2:
        display_severity_card(
            "WARNING",
            int(warning_count),
            "WARNING"
        )

    with severity_col3:
        display_severity_card(
            "CRITICAL",
            int(critical_count),
            "CRITICAL"
        )


# ==========================================
# PAGE 2 - SENSOR DATA
# ==========================================

elif page == "Sensor Data":

    st.header("Sensor Data Monitoring")

    st.subheader("Sensor Dataset")

    st.dataframe(
        data[sensor_parameters],
        use_container_width=True
    )

    st.divider()

    st.subheader("Sensor Data Visualisation")

    st.write(
        "Select the time range to monitor motor performance."
    )

    min_time = int(data["Time (seconds)"].min())
    max_time = int(data["Time (seconds)"].max())

    selected_time = st.slider(
        "Select Time Range (seconds)",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        step=5
    )

    filtered_data = data[
        (data["Time (seconds)"] >= selected_time[0]) &
        (data["Time (seconds)"] <= selected_time[1])
    ]

    st.write(
        f"Displaying readings from {selected_time[0]} "
        f"to {selected_time[1]} seconds."
    )

    st.write(
        f"Filtered Sensor Readings: {len(filtered_data)}"
    )

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:

        st.write("Temperature (°C)")

        st.line_chart(
            filtered_data,
            x="Time (seconds)",
            y="Temperature (C)"
        )

    with chart_col2:

        st.write("Vibration (mm/s)")

        st.line_chart(
            filtered_data,
            x="Time (seconds)",
            y="Vibration (mm/s)"
        )

    with chart_col3:

        st.write("Motor Speed (RPM)")

        st.line_chart(
            filtered_data,
            x="Time (seconds)",
            y="Motor Speed (RPM)"
        )


# ==========================================
# PAGE 3 - FAULT DETECTION
# ==========================================

elif page == "Fault Detection":

    st.header("Fault Detection and Severity Analysis")

    st.subheader("Fault Severity Overview")

    normal_count = (
        data["Fault Severity"] == "NORMAL"
    ).sum()

    warning_count = (
        data["Fault Severity"] == "WARNING"
    ).sum()

    critical_count = (
        data["Fault Severity"] == "CRITICAL"
    ).sum()

    severity_col1, severity_col2, severity_col3 = st.columns(3)

    with severity_col1:
        display_severity_card(
            "NORMAL",
            int(normal_count),
            "NORMAL"
        )

    with severity_col2:
        display_severity_card(
            "WARNING",
            int(warning_count),
            "WARNING"
        )

    with severity_col3:
        display_severity_card(
            "CRITICAL",
            int(critical_count),
            "CRITICAL"
        )

    st.divider()

    st.subheader("Detailed Fault Severity Report")

    severity_data = data[
        data["Fault Count"] > 0
    ]

    if not severity_data.empty:

        st.dataframe(
            severity_data[
                [
                    "Time (seconds)",
                    "Temperature (C)",
                    "Vibration (mm/s)",
                    "Motor Speed (RPM)",
                    "Fault Count",
                    "Fault Severity"
                ]
            ],
            use_container_width=True
        )

    else:

        st.success("No abnormal readings detected.")

    st.divider()

    st.subheader("Threshold-Based Fault Detection")

    st.metric(
        "Total Abnormal Readings",
        len(fault_data)
    )

    st.write("Detected Motor Faults")

    st.dataframe(
        fault_data,
        use_container_width=True
    )


# ==========================================
# PAGE 4 - ML ANOMALIES
# ==========================================

elif page == "ML Anomalies":

    st.header("Machine Learning Anomaly Detection")

    st.write(
        "Anomaly detection using the Isolation Forest algorithm."
    )

    st.metric(
        "ML Detected Anomalies",
        len(anomaly_data)
    )

    st.divider()

    st.subheader("Isolation Forest Anomaly Report")

    st.dataframe(
        anomaly_data,
        use_container_width=True
    )


# ==========================================
# PAGE 5 - STATISTICAL ANALYSIS
# ==========================================

elif page == "Statistical Analysis":

    st.header("Sensor Statistical Analysis")

    st.write(
        "Statistical summary of the motor sensor readings."
    )

    st.dataframe(
        statistics,
        use_container_width=True
    )


# ==========================================
# PAGE 6 - DOWNLOAD REPORTS
# ==========================================

elif page == "Download Reports":

    st.header("Download Project Reports")

    st.write(
        "Download the generated reports in CSV format."
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    download_col1.download_button(
        label="Download Fault Report",
        data=fault_data.to_csv(index=False).encode("utf-8"),
        file_name="fault_detection_report.csv",
        mime="text/csv"
    )

    download_col2.download_button(
        label="Download ML Anomaly Report",
        data=anomaly_data.to_csv(index=False).encode("utf-8"),
        file_name="ml_anomaly_report.csv",
        mime="text/csv"
    )

    download_col3.download_button(
        label="Download Statistical Summary",
        data=statistics.to_csv().encode("utf-8"),
        file_name="statistical_summary.csv",
        mime="text/csv"
    )


# ==========================================
# 8. DASHBOARD FOOTER
# ==========================================

st.divider()

st.caption(
    "Intelligent Predictive Maintenance System | "
    "Python | Pandas | Scikit-learn | Streamlit"
)