import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pulp
from datetime import datetime, timedelta

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="📡 Smart Bandwidth Allocation", layout="wide", initial_sidebar_state="collapsed")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #0f4c81;
        }
        h2, h3 {
            color: #1d3557;
        }
        /* Metric cards */
        .metric-card {
            background: #f8f9fa;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            margin: 10px;
        }
        .metric-card h3 {
            margin: 0;
            font-size: 22px;
            color: #0f4c81;
        }
        .metric-card p {
            font-size: 18px;
            margin: 5px 0 0 0;
        }
        /* Alerts */
        .severity-critical {
            background-color: #e63946 !important;
            color: white !important;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 6px;
        }
        .severity-warning {
            background-color: #ffb703 !important;
            color: black !important;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 6px;
        }
        /* Banner */
        .banner {
            background: linear-gradient(90deg, #0f4c81, #457b9d);
            color: white;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="banner">📡 Smart Bandwidth Allocation 🚀</div>', unsafe_allow_html=True)
st.write("")

# ---------------- Data Upload ---------------- 
uploaded_file = st.file_uploader("📂 Upload CSV (columns: tower_id, timestamp, used_mbps)", type=["csv"])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    st.success("✅ Data loaded successfully!")
else:
    st.info("Please upload a CSV file to get started.")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Live Dashboards", "🔮 Forecast", "🚨 Alerts", "⚖️ Allocation"]
)

# ---------------- Overview ----------------
with tab1:
    st.subheader("📊 Network Overview")
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><h3>🏢 Towers</h3><p>{df["tower_id"].nunique()}</p></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><h3>📑 Records</h3><p>{len(df)}</p></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><h3>📅 Date Range</h3><p>{df["timestamp"].min().date()} → {df["timestamp"].max().date()}</p></div>', unsafe_allow_html=True)

        fig = px.line(df, x="timestamp", y="used_mbps", color="tower_id",
                      title="📈 Usage by Tower", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Upload a valid CSV to view the overview.")

# ---------------- Live Dashboards ----------------
with tab2:
    st.subheader("📡 Real-Time Monitoring")
    if df is not None:
        tower = st.selectbox("Select Tower", df["tower_id"].unique())

        # Simulate live updates
        latest_time = df["timestamp"].max()
        now = datetime.now()
        time_diff = (now - latest_time).seconds // 60
        df_live = df[df["tower_id"] == tower].copy()
        df_live = df_live.tail(200)  # last 200 records

        st.markdown(f"### 📊 Live Usage Trend - {tower}")
        fig = px.line(df_live, x="timestamp", y="used_mbps",
                      title=f"Live Usage - {tower}", template="plotly_white")
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

        # Show latest usage
        current_usage = df_live["used_mbps"].iloc[-1]
        avg_usage = df_live["used_mbps"].mean()
        col1, col2 = st.columns(2)
        col1.metric("Current Usage (Mbps)", f"{current_usage:.2f}")
        col2.metric("Avg Usage (Mbps)", f"{avg_usage:.2f}")
    else:
        st.warning("⚠️ Upload a CSV to see live dashboards.")

# ---------------- Forecast ----------------
with tab3:
    st.subheader("🔮 Forecasting")
    if df is not None:
        tower = st.selectbox("Select Tower for Forecasting", df["tower_id"].unique())
        df_tower = df[df["tower_id"] == tower].sort_values("timestamp")

        df_prophet = df_tower.rename(columns={"timestamp":"ds","used_mbps":"y"})
        model_p = Prophet()
        model_p.fit(df_prophet)
        future = model_p.make_future_dataframe(periods=48, freq='H')
        forecast_p = model_p.predict(future)

        st.markdown("**📈 Prophet Forecast**")
        fig1 = model_p.plot(forecast_p)
        st.pyplot(fig1)

        st.markdown("**📉 ARIMA Forecast**")
        model_a = ARIMA(df_tower['used_mbps'], order=(2,1,2))
        model_fit = model_a.fit()
        forecast_a = model_fit.forecast(steps=48)

        plt.figure(figsize=(10,5))
        plt.plot(df_tower['timestamp'], df_tower['used_mbps'], label='Actual')
        plt.plot(pd.date_range(df_tower['timestamp'].iloc[-1], periods=48, freq='H'),
                 forecast_a, label='ARIMA Forecast')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("⚠️ Upload a CSV to generate forecasts.")

# ---------------- Alerts ----------------
with tab4:
    st.subheader("🚨 Alerts")
    if df is not None:
        capacity = st.number_input("Enter Tower Capacity (Mbps)", value=200)
        df["utilization"] = df["used_mbps"] / capacity
        alerts = df[df["utilization"] > 0.8].copy()
        alerts["severity"] = alerts["utilization"].apply(
            lambda x: "critical" if x > 0.95 else "warning"
        )

        if alerts.empty:
            st.success("✅ No alerts triggered")
        else:
            st.markdown("### ⚠️ Active Alerts")
            for _, row in alerts.iterrows():
                severity_class = "severity-critical" if row["severity"] == "critical" else "severity-warning"
                st.markdown(
                    f"<div class='{severity_class}'>Tower {row['tower_id']} | {row['timestamp']} | {row['used_mbps']} Mbps</div>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ Upload a CSV to see alerts.")

# ---------------- Allocation ----------------
with tab5:
    st.subheader("⚖️ Bandwidth Allocation Optimizer")
    if df is not None:
        latest = df.groupby("tower_id").tail(1).set_index("tower_id")
        total_bandwidth = st.number_input("Enter total available bandwidth (Mbps)", value=600)

        towers = list(latest.index)
        demand = dict(zip(towers, latest["used_mbps"]))

        prob = pulp.LpProblem("BandwidthAllocation", pulp.LpMaximize)
        alloc = pulp.LpVariable.dicts("alloc", towers, lowBound=0)

        prob += pulp.lpSum(alloc[t] for t in towers)
        prob += pulp.lpSum(alloc[t] for t in towers) <= total_bandwidth
        for t in towers:
            prob += alloc[t] <= demand[t] * 1.2
        prob.solve()

        results = {t: alloc[t].varValue for t in towers}
        st.markdown("### 📊 Optimized Allocation Results")
        st.dataframe(pd.DataFrame(results.items(), columns=["Tower","Allocated_Mbps"]))
    else:
        st.warning("⚠️ Upload a CSV to optimize bandwidth.")
