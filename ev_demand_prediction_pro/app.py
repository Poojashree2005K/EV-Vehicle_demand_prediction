import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# === Page Config ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load Model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Dark Neon Styling ===
st.markdown("""
    <style>
        .stApp {
            background-color: #0c0c0c;
            color: #00FFCC;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FFCC;
        }
        .stButton>button {
            background-color: #00FFCC;
            color: #0c0c0c;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #00e6b8;
        }
        .stSelectbox>div>div, .stMultiSelect>div>div, .stNumberInput input {
            background-color: #1c1c1c;
            color: #00FFCC;
        }
    </style>
""", unsafe_allow_html=True)

# === Title and Image ===
st.markdown("""
    <div style='text-align: center; font-size: 40px; font-weight: bold; color: #00FFCC; margin-top: 20px;'>
        🔋 EV Growth Forecasting Dashboard for Washington State Counties
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: normal; color: #00FFCC; margin-bottom: 25px;'>
        Visualize and Compare Predicted Electric Vehicle Adoption Across Regions
    </div>
""", unsafe_allow_html=True)

st.image("evc.png", use_container_width=True)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique().tolist())

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["🔍 Forecast", "📊 Compare Counties", "ℹ️ About"])

# ========== TAB 1: Forecast Single County ==========
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        county = st.selectbox("Select a County", county_list)
    with col2:
        forecast_year = st.number_input(
            "Target Forecast Year", 
            min_value=datetime.now().year, 
            max_value=2050, 
            value=datetime.now().year + 3,
            step=1
        )

    county_df = df[df['County'] == county].sort_values("Date")
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    target_date = pd.Timestamp(f"{forecast_year}-12-31")
    months_diff = (target_date.year - latest_date.year) * 12 + (target_date.month - latest_date.month)
    forecast_horizon = max(months_diff, 1)

    future_rows = []
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) >= 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    historical_cum['Source'] = 'Historical'
    historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()
    forecast_df = pd.DataFrame(future_rows)
    forecast_df['Source'] = 'Forecast'
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]
    combined = pd.concat([
        historical_cum[['Date', 'Cumulative EV', 'Source']],
        forecast_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)

    st.subheader(f"📊 Cumulative EV Forecast for {county} County (up to {forecast_year})")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, data in combined.groupby('Source'):
        ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
    ax.set_title(f"Cumulative EV Trend - {county} (Forecast through {forecast_year})", fontsize=14, color='#00FFCC')
    ax.set_xlabel("Date", color='#00FFCC')
    ax.set_ylabel("Cumulative EV Count", color='#00FFCC')
    ax.grid(True, color='#333333', alpha=0.5)
    ax.set_facecolor("#0c0c0c")
    fig.patch.set_facecolor('#0c0c0c')
    ax.tick_params(colors='#00FFCC')
    ax.legend()
    st.pyplot(fig)

    historical_total = historical_cum['Cumulative EV'].iloc[-1]
    forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
    if historical_total > 0:
        forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
        trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
        st.markdown(f"""
            <div style='color:#00FFCC; font-size:18px;'>
            ✅ EV adoption in <b>{county}</b> is expected to show a <b>{trend} of {forecast_growth_pct:.2f}%</b> by {forecast_year}.
            </div>
        """, unsafe_allow_html=True)

# ========== TAB 2: Compare Counties ==========
with tab2:
    st.subheader(f"📊 Compare EV Adoption Trends (Up to {forecast_year})")
    multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)
    if multi_counties:
        comparison_data = []
        for cty in multi_counties:
            cty_df = df[df['County'] == cty].sort_values("Date")
            cty_code = cty_df['county_encoded'].iloc[0]
            hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = cty_df['months_since_start'].max()
            last_date = cty_df['Date'].max()
            future_rows_cty = []

            for i in range(1, months_diff + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                ev_slope = np.polyfit(range(len(cum_ev[-6:])), cum_ev[-6:], 1)[0] if len(cum_ev) >= 6 else 0

                new_row = {
                    'months_since_start': months_since,
                    'county_encoded': cty_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }

                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)
                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)

            hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
            fc_df = pd.DataFrame(future_rows_cty)
            fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
            combined_cty = pd.concat([hist_cum[['Date', 'Cumulative EV']], fc_df[['Date', 'Cumulative EV']]], ignore_index=True)
            combined_cty['County'] = cty
            comparison_data.append(combined_cty)

        comp_df = pd.concat(comparison_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in comp_df.groupby('County'):
            ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
        ax.set_title(f"EV Adoption Forecast Comparison Through {forecast_year}", fontsize=16, color='#00FFCC')
        ax.set_xlabel("Date", color='#00FFCC')
        ax.set_ylabel("Cumulative EV Count", color='#00FFCC')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#0c0c0c")
        fig.patch.set_facecolor('#0c0c0c')
        ax.tick_params(colors='#00FFCC')
        ax.legend(title="County")
        st.pyplot(fig)

        growth_summaries = []
        for cty in multi_counties:
            cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
            historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - months_diff - 1]
            forecasted_total = cty_df['Cumulative EV'].iloc[-1]
            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
            else:
                growth_summaries.append(f"{cty}: N/A")

        st.markdown(f"""
            <div style='color:#00FFCC; font-size:18px;'>
            📈 Forecasted EV Growth by {forecast_year}: {" | ".join(growth_summaries)}
            </div>
        """, unsafe_allow_html=True)

# ========== TAB 3: About ==========
with tab3:
    st.subheader("ℹ️ About This Application")
    st.markdown("""
    Welcome to the **EV Adoption Forecasting App** – a smart analytics tool designed to predict and visualize the electric vehicle (EV) growth trends across counties in **Washington State**.

    **🔍 Features:**
    - Select any county and forecast EV adoption up to a chosen year.
    - Compare EV growth across multiple counties with visually rich graphs.
    - Interactive interface with a futuristic dark neon theme for enhanced readability and aesthetics.

    **🧠 Powered By:**
    - Machine Learning Regression Model (pre-trained)
    - Python, Streamlit, Pandas, NumPy, and Matplotlib

    **🎯 Purpose:**
    - Designed to support policy makers, researchers, and sustainability advocates by providing predictive insights on electric vehicle trends.

    **📌 Built For:**
    - Developed to promote data-driven decision making for sustainable mobility planning

    ---
    🔗 **Future Enhancements Coming Soon:**
    - Downloadable reports
    - Geo-spatial visualization using interactive maps
    - More granular insights based on vehicle types or demographics

    Stay tuned! ⚡️🚗
    """)
