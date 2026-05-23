"""
Drift monitoring page for IndustrialSentinel dashboard.
"""
import streamlit as st
import pandas as pd
import requests
import os
import json

st.set_page_config(page_title="Drift Monitor", page_icon="📡", layout="wide")
st.title("📡 Drift Monitor")

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = os.environ.get("DATA_DIR", "data")

st.markdown("Submit engine data to check for sensor drift against training distribution.")

# Load sample test data for demo
@st.cache_data(ttl=60)
def load_test_sample():
    """Load a sample from test data for drift demo."""
    raw_path = f"{DATA_DIR}/raw/test_FD001.txt"
    try:
        cols = [f"col{i}" for i in range(26)]
        df = pd.read_csv(raw_path, sep=r'\s+', header=None, names=cols)
        return df
    except FileNotFoundError:
        return None


test_data = load_test_sample()

if test_data is not None:
    engine_id = st.selectbox("Select test engine for drift analysis", range(1, 101))

    if st.button("Run Drift Detection"):
        # Get engine data (sensor columns are indices 5-25)
        engine_rows = test_data[test_data["col0"] == engine_id]
        readings = engine_rows.iloc[:, 5:26].values.tolist()

        try:
            resp = requests.post(
                f"{API_URL}/drift",
                json={"unit": engine_id, "readings": readings},
                timeout=30
            )
            if resp.status_code == 200:
                result = resp.json()
                data = result["data"]

                st.subheader("Drift Detection Results")
                col1, col2 = st.columns(2)
                col1.metric("Sensors Checked", data["sensors_checked"])
                col2.metric("Sensors Drifted", data["sensors_drifted"])

                if data["drifted_sensor_names"]:
                    st.warning(f"Drifted sensors: {', '.join(data['drifted_sensor_names'])}")
                else:
                    st.success("No drift detected!")

                # Per-sensor bar chart
                st.subheader("KS Statistics by Sensor")
                per_sensor = data["per_sensor"]
                chart_data = pd.DataFrame({
                    "Sensor": list(per_sensor.keys()),
                    "KS Statistic": [v["ks_statistic"] for v in per_sensor.values()],
                    "P-Value": [v["p_value"] for v in per_sensor.values()],
                })
                chart_data = chart_data.set_index("Sensor")
                st.bar_chart(chart_data["KS Statistic"])

                # Threshold line info
                st.caption("Red threshold: p-value < 0.05 indicates drift")

                # Detailed table
                st.subheader("Detailed Report")
                detail_df = pd.DataFrame([
                    {"Sensor": k, "KS Stat": v["ks_statistic"],
                     "P-Value": v["p_value"], "Drifted": v["drift_detected"]}
                    for k, v in per_sensor.items()
                ])
                st.dataframe(detail_df, use_container_width=True)
            else:
                st.error(f"API error: {resp.status_code} - {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the API service is running.")
else:
    st.error("Test data not found.")
