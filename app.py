import pandas as pd
import streamlit as st

DATA_PATH = "data/pakistan_rainfall_sample.csv"

st.set_page_config(page_title="Pakistan Rainfall Scenarios", layout="wide")

st.title("Pakistan Rainfall Scenario Explorer")
st.write(
    "Explore historical rainfall and future projections for major Pakistani cities. "
    "Use the sidebar filters to compare scenarios across time."
)

@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m")
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error("Sample data not found. Run `python scripts/make_sample_data.py` first.")
    st.stop()

st.sidebar.header("Filters")
scenarios = st.sidebar.multiselect(
    "Scenario",
    options=sorted(df["scenario"].unique()),
    default=sorted(df["scenario"].unique()),
)
locations = st.sidebar.multiselect(
    "Location",
    options=sorted(df["district_or_city"].unique()),
    default=sorted(df["district_or_city"].unique()),
)

min_date = df["date"].min().to_pydatetime()
max_date = df["date"].max().to_pydatetime()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

filtered = df[
    df["scenario"].isin(scenarios)
    & df["district_or_city"].isin(locations)
    & (df["date"] >= pd.to_datetime(start_date))
    & (df["date"] <= pd.to_datetime(end_date))
]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

summary = (
    filtered.groupby(["scenario", "province", "district_or_city"], as_index=False)[
        "rainfall_mm"
    ]
    .mean()
    .sort_values(["scenario", "province", "district_or_city"])
)

st.subheader("Average Monthly Rainfall (mm)")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Monthly Rainfall Over Time")
chart_data = (
    filtered.groupby(["date", "scenario", "district_or_city"], as_index=False)[
        "rainfall_mm"
    ]
    .mean()
)

for scenario in scenarios:
    scenario_data = chart_data[chart_data["scenario"] == scenario]
    st.markdown(f"**{scenario.upper()}**")
    pivot = scenario_data.pivot(
        index="date", columns="district_or_city", values="rainfall_mm"
    ).sort_index()
    st.line_chart(pivot)

st.caption("Sample data is synthetic and for demonstration only.")
