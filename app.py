import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = "data/pakistan_rainfall_sample.csv"
REQUIRED_COLUMNS = {
    "date",
    "province",
    "district_or_city",
    "scenario",
    "rainfall_mm",
}
TEMPLATE_COLUMNS = [
    "date",
    "province",
    "district_or_city",
    "scenario",
    "rainfall_mm",
]

st.set_page_config(page_title="Pakistan Rainfall Scenarios", layout="wide")

st.title("Pakistan Rainfall Scenario Explorer")
st.write(
    "Explore historical rainfall and future projections for major Pakistani cities. "
    "Use the sidebar filters to compare scenarios across time."
)

@st.cache_data
def load_data_from_csv(file_or_path) -> pd.DataFrame:
    data = pd.read_csv(file_or_path)
    data.columns = [column.strip() for column in data.columns]
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}.")

    date_series = data["date"]
    try:
        parsed_dates = pd.to_datetime(date_series, infer_datetime_format=True, errors="raise")
    except (ValueError, TypeError):
        parsed_ym = pd.to_datetime(date_series, format="%Y-%m", errors="coerce")
        parsed_ymd = pd.to_datetime(date_series, format="%Y-%m-%d", errors="coerce")
        parsed_dates = parsed_ym.fillna(parsed_ymd)

    if parsed_dates.isna().any():
        raise ValueError(
            "Column 'date' must use YYYY-MM or YYYY-MM-DD format and be parseable."
        )

    data["date"] = parsed_dates
    data["rainfall_mm"] = pd.to_numeric(data["rainfall_mm"], errors="coerce")
    if data["rainfall_mm"].isna().any():
        raise ValueError("Column 'rainfall_mm' must be numeric.")

    return data


st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Choose a data source",
    options=("Use sample data", "Upload my CSV"),
    index=0,
)

template_rows = [
    ["1980-01", "Sindh", "Karachi", "historical", 12.5],
    ["2050-07-01", "Punjab", "Lahore", "ssp245", 180.2],
]
template_df = pd.DataFrame(template_rows, columns=TEMPLATE_COLUMNS)
st.sidebar.download_button(
    "Download template CSV",
    data=template_df.to_csv(index=False),
    file_name="rainfall_template.csv",
    mime="text/csv",
)

try:
    if data_source == "Upload my CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )
        if uploaded_file is None:
            st.info("Upload a CSV file to continue.")
            st.stop()
        df = load_data_from_csv(uploaded_file)
    else:
        df = load_data_from_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Sample data not found. Run `python scripts/make_sample_data.py` first.")
    st.stop()
except ValueError as error:
    st.error(str(error))
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

st.sidebar.subheader("Seasonal view")
monsoon_only = st.sidebar.toggle("Monsoon season only (Jul-Sep)", value=False)

filtered = filtered.copy()
filtered["month"] = filtered["date"].dt.month
filtered_view = filtered.copy()
if monsoon_only:
    filtered_view = filtered_view[filtered_view["month"].between(7, 9)]
    if filtered_view.empty:
        st.warning("No data for the selected filters in monsoon months.")
        st.stop()

st.subheader("Summary Statistics")
total_rainfall = filtered_view["rainfall_mm"].sum()
avg_rainfall = filtered_view["rainfall_mm"].mean()
max_rainfall = filtered_view["rainfall_mm"].max()
min_rainfall = filtered_view["rainfall_mm"].min()

summary_cols = st.columns(4)
summary_cols[0].metric("Total rainfall (mm)", f"{total_rainfall:,.1f}")
summary_cols[1].metric("Average rainfall (mm)", f"{avg_rainfall:,.1f}")
summary_cols[2].metric("Maximum rainfall (mm)", f"{max_rainfall:,.1f}")
summary_cols[3].metric("Minimum rainfall (mm)", f"{min_rainfall:,.1f}")

st.subheader("Monthly Rainfall Over Time")
line_chart = (
    alt.Chart(filtered_view)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rainfall_mm:Q", title="Rainfall (mm)"),
        color=alt.Color("scenario:N", title="Scenario"),
        detail="district_or_city:N",
        tooltip=[
            alt.Tooltip("district_or_city:N", title="City"),
            alt.Tooltip("scenario:N", title="Scenario"),
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("rainfall_mm:Q", title="Rainfall (mm)", format=".1f"),
        ],
    )
    .properties(height=320)
)
st.altair_chart(line_chart, use_container_width=True)

st.subheader("Monthly Climatology")
climatology = (
    filtered_view.groupby(["scenario", "month"], as_index=False)["rainfall_mm"]
    .mean()
    .sort_values(["month", "scenario"])
)
climatology_chart = (
    alt.Chart(climatology)
    .mark_line(point=True)
    .encode(
        x=alt.X("month:O", title="Month"),
        y=alt.Y("rainfall_mm:Q", title="Average rainfall (mm)"),
        color=alt.Color("scenario:N", title="Scenario"),
        tooltip=[
            alt.Tooltip("scenario:N", title="Scenario"),
            alt.Tooltip("month:O", title="Month"),
            alt.Tooltip("rainfall_mm:Q", title="Average rainfall (mm)", format=".1f"),
        ],
    )
    .properties(height=280)
)
st.altair_chart(climatology_chart, use_container_width=True)

st.subheader("Monsoon Analysis (Jul-Sep)")
monsoon_data = filtered[filtered["month"].between(7, 9)]
if monsoon_data.empty:
    st.info("No monsoon-season data for the selected filters.")
else:
    monsoon_total = monsoon_data["rainfall_mm"].sum()
    monsoon_avg = monsoon_data["rainfall_mm"].mean()
    monsoon_cols = st.columns(2)
    monsoon_cols[0].metric("Monsoon total rainfall (mm)", f"{monsoon_total:,.1f}")
    monsoon_cols[1].metric("Monsoon average rainfall (mm)", f"{monsoon_avg:,.1f}")

    scenario_order = ["historical", "ssp245", "ssp585"]
    available_scenarios = [s for s in scenario_order if s in monsoon_data["scenario"].unique()]
    monsoon_summary = (
        monsoon_data.groupby("scenario", as_index=False)["rainfall_mm"]
        .agg(total_rainfall="sum", average_rainfall="mean")
        .sort_values(
            "scenario",
            key=lambda series: pd.Categorical(
                series, categories=available_scenarios, ordered=True
            ),
        )
    )
    monsoon_totals_chart = (
        alt.Chart(monsoon_summary)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", title="Scenario"),
            y=alt.Y("total_rainfall:Q", title="Total rainfall (mm)"),
            color=alt.Color("scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("total_rainfall:Q", title="Total rainfall (mm)", format=".1f"),
            ],
        )
        .properties(height=240)
    )
    monsoon_avg_chart = (
        alt.Chart(monsoon_summary)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", title="Scenario"),
            y=alt.Y("average_rainfall:Q", title="Average rainfall (mm)"),
            color=alt.Color("scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip(
                    "average_rainfall:Q", title="Average rainfall (mm)", format=".1f"
                ),
            ],
        )
        .properties(height=240)
    )
    monsoon_chart_cols = st.columns(2)
    with monsoon_chart_cols[0]:
        st.altair_chart(monsoon_totals_chart, use_container_width=True)
    with monsoon_chart_cols[1]:
        st.altair_chart(monsoon_avg_chart, use_container_width=True)

summary = (
    filtered_view.groupby(["scenario", "province", "district_or_city"], as_index=False)[
        "rainfall_mm"
    ]
    .mean()
    .sort_values(["scenario", "province", "district_or_city"])
)

st.subheader("Average Monthly Rainfall (mm)")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.caption("Sample data is synthetic and for demonstration only.")
