import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import pydeck as pdk
import os

def dashboard_energy_map():

    with st.spinner("Fetching data..."):
        placeholder = st.empty()
        placeholder.markdown("<h2 style='text-align:center'>Fetching data...</h2>", unsafe_allow_html=True)

        # ---- CONFIG ----
        st.set_page_config(page_title="Italy Energy Dashboard", layout="wide")
        API_TOKEN = os.environ["ELECTRICITYMAPS_TOKEN"]
        ZONE = "IT"  # Italy
        headers = {"auth-token": API_TOKEN}

        # ---- FETCH FUNCTIONS ----
        def get_latest_carbon_intensity(zone):
            url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"Carbon intensity API error {r.status_code}: {r.text}")
                return None

        def get_latest_power_breakdown(zone):
            url = f"https://api.electricitymap.org/v3/power-breakdown/latest?zone={zone}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"Power breakdown API error {r.status_code}: {r.text}")
                return None

        # ---- FETCH DATA ----
        carbon_data = get_latest_carbon_intensity(ZONE)
        power_data = get_latest_power_breakdown(ZONE)

    with st.spinner("Processing data..."):
        placeholder.markdown("<h2 style='text-align:center'>Processing data...</h2>", unsafe_allow_html=True)

        # ---- PROCESS DATA WITH PANDAS ----
        if power_data and "powerConsumptionBreakdown" in power_data:
            mix = power_data["powerConsumptionBreakdown"]
            df_pd = pd.DataFrame(list(mix.items()), columns=["Source", "MW"])
            total_mw = df_pd["MW"].sum()
            df_pd["Percent"] = (df_pd["MW"] / total_mw * 100).round(1)

            # Define renewable vs fossil
            renewables = ["wind", "solar", "hydro", "geothermal", "biomass"]
            df_pd["Type"] = df_pd["Source"].apply(lambda x: "Renewable" if x in renewables else "Fossil")
            df_plot = df_pd
        else:
            df_plot = pd.DataFrame(columns=["Source", "MW", "Percent", "Type"])
            total_mw = 0

        # ----------------------------
        # Extract datetime
        # ----------------------------
        dt = datetime.fromisoformat(power_data["datetime"].replace("Z", "+00:00"))
        placeholder.empty()

        # ---- STREAMLIT LAYOUT ----
        st.title(f"Italian Electricity Mix — {dt.strftime('%Y-%m-%d %H:%M UTC')}")
        st.subheader("sourced from: api.electricitymap.org")

        # ---- KPI ROW ----
        kpi1, kpi2, kpi3 = st.columns(3)
        if carbon_data:
            intensity = carbon_data.get("carbonIntensity")
            updated_at = carbon_data.get("datetime")
            kpi1.metric("Carbon Intensity (gCO₂/kWh)", intensity, help=f"Updated at {updated_at}")

        kpi2.metric("Total Consumption (MW)", f"{total_mw:.0f}")

        if not df_plot.empty:
            top_source = df_plot.sort_values("MW", ascending=False).iloc[0]["Source"]
            kpi3.metric("Top Source", top_source)

        # ---- CHARTS ----
        st.subheader("Power Mix")
        col1, col2, col3 = st.columns(3)

        with col1:
            if not df_plot.empty:
                fig = px.pie(df_plot, values="Percent", names="Source", title="Power Breakdown (%)")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not df_plot.empty:
                df_type = df_plot.groupby("Type")["MW"].sum().reset_index()
                fig3 = px.pie(df_type, values="MW", names="Type", title="Renewables vs Fossil Share")
                st.plotly_chart(fig3, use_container_width=True)

        with col3:
            if not df_plot.empty:
                fig2 = px.bar(df_plot, x="Source", y="MW", color="Type", title="Power Breakdown (MW) by Source")
                st.plotly_chart(fig2, use_container_width=True)

        # ----------------------------
        # Import/Export Map
        # ----------------------------
        imports = power_data.get("powerImportBreakdown", {})
        exports = power_data.get("powerExportBreakdown", {})

        coords = {
            "IT": [12.5, 41.9],  # Rome center
            "FR": [2.5, 46.5],
            "FR-COR": [9.0, 42.0],
            "CH": [8.2, 46.8],
            "AT": [14.5, 47.5],
            "SI": [14.5, 46.1],
            "ME": [19.4, 42.7],
            "MT": [14.5, 35.9],
        }

        flows = []
        for country, val in imports.items():
            if val > 0 and country in coords:
                flows.append({"start": coords[country], "end": coords["IT"], "value": val, "type": "Import"})
        for country, val in exports.items():
            if val > 0 and country in coords:
                flows.append({"start": coords["IT"], "end": coords[country], "value": val, "type": "Export"})

        flow_df = pd.DataFrame(flows)
        if not flow_df.empty:
            st.subheader("Cross-Border Electricity Flows")
            layer = pdk.Layer(
                "ArcLayer",
                data=flow_df,
                get_source_position="start",
                get_target_position="end",
                get_width="value/200",
                get_tilt=15,
                get_source_color=[0, 128, 255],
                get_target_color=[255, 0, 0],
                pickable=True,
                auto_highlight=True,
            )
            view_state = pdk.ViewState(latitude=42.5, longitude=12.5, zoom=4.5, bearing=0, pitch=30)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{type} {value} MW"})
            st.pydeck_chart(r)

        # ---- 24h HISTORY ----
        history_placeholder = st.empty()
        history_placeholder2 = st.empty()
        history_placeholder3 = st.empty()

        with st.spinner("Fetching 24h history..."):

            @st.cache_data
            def get_power_history(zone="IT"):
                url = f"https://api.electricitymap.org/v3/power-breakdown/history?zone={zone}"
                r = requests.get(url, headers=headers)
                if r.status_code == 200:
                    return pd.DataFrame(r.json().get("history", []))
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
                    return pd.DataFrame()

            df = get_power_history(ZONE)

            if not df.empty:
                breakdowns = df["powerConsumptionBreakdown"].apply(pd.Series)
                df = pd.concat([df[["datetime"]], breakdowns], axis=1)
                df["datetime"] = pd.to_datetime(df["datetime"])

                # ---- TIME SERIES ----
                history_placeholder.subheader("Last 24h Consumption Breakdown")
                fig = px.area(df, x="datetime", y=["solar", "wind", "hydro", "gas", "coal"],
                              title="Electricity Mix Over Time",
                              labels={"value": "MW", "datetime": "Time"})
                history_placeholder2.plotly_chart(fig, use_container_width=True)

                # ---- RENEWABLES VS FOSSIL ----
                renewables = ["solar", "wind", "hydro", "geothermal", "biomass"]
                df["renewable"] = df[renewables].sum(axis=1)
                df["fossil"] = df[["gas", "coal", "oil"]].sum(axis=1)
                fig2 = px.area(df, x="datetime", y=["renewable", "fossil"], title="Renewables vs Fossil Over Time")
                history_placeholder3.plotly_chart(fig2, use_container_width=True)

            # ---- RAW DATA ----
            with st.expander("Raw API Data"):
                st.json({"carbon": carbon_data, "power": power_data})


def dashboard_finance():
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import plotly.express as px
    from prophet import Prophet

    # ---------------------------
    # Available tickers
    tickers = {
        "Commodities": {
            "Wheat": "ZW=F",
            "Corn": "ZC=F",
            "Soybeans": "ZS=F",
            "Coffee": "KC=F",
            "Rice": "ZR=F",
            "Gold": "GC=F",
            "Silver": "SI=F",
            "Crude Oil": "CL=F",
            "Natural Gas": "NG=F",
        },
        "Currencies": {
            "USD/JPY": "JPY=X",
            "GBP/USD": "GBPUSD=X",
            "EUR/USD": "EURUSD=X",
        },
        "Indices": {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
        }
    }

    st.set_page_config(page_title="Finance Dashboard", layout="wide")
    # ---------------------------
    # Sidebar selection
    category = st.sidebar.selectbox("Select Category", list(tickers.keys()))
    ticker_name = st.sidebar.selectbox("Select Ticker", list(tickers[category].keys()))
    ticker = tickers[category][ticker_name]

    # ---------------------------
    st.title(f"Finance Dashboard — {ticker_name}")

    # ---------------------------
    # Fetch data
    with st.spinner(f"Downloading {ticker_name} data..."):
        data = yf.download(ticker, period="5y", interval="1d").reset_index()

    # Flatten multi-index columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in data.columns]

    # Find the correct Close column
    close_col = [col for col in data.columns if col.startswith("Close")][0]


    # ---------------------------
    # Prophet Forecast

    # Prepare data for Prophet
    df_prophet = data[["Date", close_col]].rename(columns={"Date": "ds", close_col: "y"})
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
    df_prophet = df_prophet.dropna()

    # Fit model
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,changepoint_prior_scale=0.01)  # smaller = smoother trend
    m.fit(df_prophet)

    # Make future dataframe

    period_extrapolation = st.slider("", 1, 1000, 365)
    st.subheader(f"Forecast with Prophet ({period_extrapolation} days)")

    future = m.make_future_dataframe(periods=period_extrapolation,freq='d')
    forecast = m.predict(future)

    # Only future dates (after the last historical date)
    last_date = df_prophet["ds"].iloc[-3]
    # st.write(df_prophet["ds"].iloc[-2:])
    forecast_future = forecast[forecast["ds"] > last_date]
    forecast_future["yhat_upper"].iloc[0] = forecast_future["ds"].iloc[0]
    forecast_future["yhat_lower"].iloc[0] = forecast_future["ds"].iloc[0]

    # Plot historical + forecast
    import plotly.graph_objects as go

    fig_forecast = go.Figure()

    # Historical prices
    fig_forecast.add_trace(go.Scatter(
        x=df_prophet["ds"], y=df_prophet["y"],
        mode="lines", name="Historical",
        # line=dict(color="steelblue")
    ))

    # Forecasted prices
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="lightcoral", dash="dash"),

    ))

    # Forecast uncertainty band
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat_upper"],
        mode="lines", name="Upper", line=dict(color="lightcoral"), showlegend=False
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat_lower"],
        mode="lines", name="Lower", line=dict(color="lightcoral"), fill="tonexty", fillcolor="rgba(255,0,0,0.2)",
        showlegend=False
    ))

    fig_forecast.update_layout(
        title=f"{ticker_name} Forecast (Prophet)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # ---------------------------
    # Show raw data if needed
    with st.expander("Show Raw Data"):
        st.dataframe(data)

    # # ---------------------------
    # # Price chart
    # st.subheader("Historical Prices")
    # fig_price = px.line(
    #     data,
    #     x="Date",
    #     y=close_col,
    #     title=f"{ticker_name} Closing Prices",
    #     labels={close_col: "Price", "Date": "Date"}
    # )
    # st.plotly_chart(fig_price, use_container_width=True)


    # ---------------------------
    # Additional analysis

    st.subheader("Moving Average & Rolling Volatility")

    # User-adjustable window
    ma_window = st.slider("MA / Volatility Window (days)", 1, 100, 30)

    data['MA'] = data[close_col].rolling(ma_window).mean()
    data['Volatility'] = data[close_col].rolling(ma_window).std()

    # ---------------------------
    # Simple anomaly detection: +/- n std
    st.subheader("Anomaly Detection")

    anomaly_std = st.slider("Anomaly Threshold (std)", 1.0, 5.0, 2.5)
    data['Z_score'] = (data[close_col] - data['MA']) / data['Volatility']
    data['Anomaly'] = data['Z_score'].abs() > anomaly_std

    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(go.Scatter(x=data["Date"], y=data[close_col], mode="lines", name="Close"))
    fig_anomaly.add_trace(go.Scatter(
        x=data.loc[data['Anomaly'], "Date"],
        y=data.loc[data['Anomaly'], close_col],
        mode="markers", name="Anomaly", marker=dict(color="red", size=8)
    ))
    fig_anomaly.add_trace(go.Scatter(x=data["Date"], y=data['MA'], mode="lines", name=f"{ma_window}-day MA", line=dict(color="orange")))

    fig_anomaly.update_layout(title=f"{ticker_name} Anomalies (±{anomaly_std}σ)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_anomaly, use_container_width=True)

    # # Price + MA
    # fig_ma = go.Figure()
    # fig_ma.add_trace(go.Scatter(x=data["Date"], y=data[close_col], mode="lines", name="Close"))
    # fig_ma.add_trace(go.Scatter(x=data["Date"], y=data['MA'], mode="lines", name=f"{ma_window}-day MA", line=dict(color="orange")))

    # fig_ma.update_layout(title=f"{ticker_name} Price & Moving Average", xaxis_title="Date", yaxis_title="Price")
    # st.plotly_chart(fig_ma, use_container_width=True)

    # Volatility
    fig_vol = px.line(data, x="Date", y="Volatility", title=f"{ticker_name} Rolling Volatility ({ma_window}-day)")
    st.plotly_chart(fig_vol, use_container_width=True)

    # ---------------------------
    # Volume chart
    volume_col = [col for col in data.columns if col.startswith("Volume")][0]
    st.subheader("Trading Volume")
    fig_volume = px.bar(
        data,
        x="Date",
        y=volume_col,
        title=f"{ticker_name} Volume",
        labels={volume_col: "Volume", "Date": "Date"}
    )
    st.plotly_chart(fig_volume, use_container_width=True)




# ---- SIDEBAR ----

from PIL import Image
# Load image
# Display in sidebar
st.sidebar.image("assets/avatar_open_to_work.png", caption="Sébastien MARTINET", use_container_width=True)

choice = st.sidebar.selectbox("Select Dashboard", ["Italian Energy Production", "Finance forecast"], index=0)

if choice == "Italian Energy Production":
    dashboard_energy_map()
else:
    dashboard_finance()
