"""
Drift monitoring page for IndustrialSentinel dashboard.
Detect sensor distribution shifts that may degrade model performance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="Drift Monitor", page_icon="📡", layout="wide")
st.title("📡 Sensor Drift Monitor")
st.markdown("Detect distribution shifts in sensor data that may degrade model accuracy")
st.markdown("---")

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = os.environ.get("DATA_DIR", "data")

st.markdown("""
**What is drift?** When the statistical distribution of incoming sensor data differs significantly 
from the training data, model predictions become unreliable. This page uses the Kolmogorov-Smirnov 
test to detect per-sensor drift.
""")

st.markdown("---")


@st.cache_data(ttl=60)
def load_test_data():
    """Load test data for drift demo."""
    raw_path = f"{DATA_DIR}/raw/test_FD001.txt"
    try:
        cols = [f"col{i}" for i in range(26)]
        return pd.read_csv(raw_path, sep=r'\s+', header=None, names=cols)
    except FileNotFoundError:
        return None


test_data = load_test_data()

if test_data is not None:
    st.markdown("### Run Drift Detection")

    col1, col2 = st.columns([1, 2])
    with col1:
        engine_id = st.selectbox("Select test engine", range(1, 101))
        run_button = st.button("🔍 Analyze Drift", type="primary", use_container_width=True)

    if run_button:
        engine_rows = test_data[test_data["col0"] == engine_id]
        readings = engine_rows.iloc[:, 5:26].values.tolist()

        with st.spinner("Running KS tests against training distribution..."):
            try:
                resp = requests.post(
                    f"{API_URL}/drift",
                    json={"unit": engine_id, "readings": readings},
                    timeout=30
                )

                if resp.status_code == 200:
                    result = resp.json()
                    data = result["data"]

                    # Summary
                    st.markdown("---")
                    st.markdown("### Results")

                    sum_col1, sum_col2, sum_col3 = st.columns(3)
                    sum_col1.metric("Sensors Checked", data["sensors_checked"])
                    sum_col2.metric("Sensors Drifted", data["sensors_drifted"])
                    drift_pct = data["sensors_drifted"] / data["sensors_checked"] * 100
                    sum_col3.metric("Drift Percentage", f"{drift_pct:.0f}%")

                    if data["sensors_drifted"] == 0:
                        st.success("✅ No drift detected! Sensor distributions match training data.")
                    elif data["sensors_drifted"] <= 3:
                        st.warning(f"⚠️ Minor drift detected in {data['sensors_drifted']} sensors: {', '.join(data['drifted_sensor_names'])}")
                    else:
                        st.error(f"🚨 Significant drift detected in {data['sensors_drifted']} sensors! Model predictions may be unreliable.")

                    st.markdown("---")

                    # Per-sensor results
                    st.markdown("### Per-Sensor KS Statistics")
                    per_sensor = data["per_sensor"]

                    chart_df = pd.DataFrame({
                        "Sensor": list(per_sensor.keys()),
                        "KS Statistic": [v["ks_statistic"] for v in per_sensor.values()],
                        "P-Value": [v["p_value"] for v in per_sensor.values()],
                        "Drifted": ["🔴 Yes" if v["drift_detected"] else "🟢 No" for v in per_sensor.values()],
                    })

                    # Bar chart
                    bar_data = chart_df.set_index("Sensor")["KS Statistic"]
                    st.bar_chart(bar_data, height=300)
                    st.caption("Higher KS statistic = greater distribution difference. Threshold: p-value < 0.05")

                    # Detailed table
                    st.markdown("### Detailed Report")
                    st.dataframe(chart_df, use_container_width=True, hide_index=True)

                    # Explanation
                    with st.expander("ℹ️ How to interpret"):
                        st.markdown("""
                        - **KS Statistic**: Maximum difference between cumulative distributions (0-1). Higher = more different.
                        - **P-Value**: Probability that the two samples come from the same distribution. Below 0.05 = statistically significant drift.
                        - **Action**: If many sensors drift, retrain the model on recent data or investigate operational changes.
                        """)
                else:
                    st.error(f"API error: {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the API service is running.")
else:
    st.error("⚠️ Test data not found.")
