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
    st.write("Finance dashboard placeholder")


# ---- SIDEBAR ----
choice = st.sidebar.selectbox("Select Dashboard", ["Italian Energy Production", "Finance forecast"], index=0)

if choice == "Italian Energy Production":
    dashboard_energy_map()
else:
    dashboard_finance()
