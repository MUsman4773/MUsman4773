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

DATE_FORMATS = ["%Y-%m", "%Y-%m-%d", "%Y/%m/%d"]


def parse_dates(date_series: pd.Series) -> pd.Series:
    cleaned = date_series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=cleaned.index)
    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(cleaned, format=fmt, errors="coerce"))
    if parsed.isna().any():
        raise ValueError(
            "Column 'date' must use YYYY-MM, YYYY-MM-DD, or YYYY/MM/DD formats "
            "(e.g., 1980-01, 1980-01-15, 1980/01/15)."
        )
    return parsed


def normalize_schema(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    rename_map = {}
    if "district_or_city" not in data.columns and "city" in data.columns:
        rename_map["city"] = "district_or_city"
    if "rainfall_mm" not in data.columns and "rainfall" in data.columns:
        rename_map["rainfall"] = "rainfall_mm"
    if rename_map:
        data = data.rename(columns=rename_map)
    return data


@st.cache_data
def load_data_from_csv(file_or_path) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(file_or_path)
    data.columns = [column.strip() for column in data.columns]
    data = normalize_schema(data)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        template_schema = ", ".join(TEMPLATE_COLUMNS)
        raise ValueError(
            "Missing required columns: "
            f"{missing_list}. Expected schema: {template_schema}."
        )

    data["date"] = parse_dates(data["date"])
    data["rainfall_mm"] = pd.to_numeric(data["rainfall_mm"], errors="coerce")
    invalid_rainfall = data["rainfall_mm"].isna().sum()
    warnings = []
    if invalid_rainfall:
        data = data.dropna(subset=["rainfall_mm"])
        warnings.append(
            f"Dropped {invalid_rainfall:,} row(s) with non-numeric rainfall_mm values."
        )

    return data, warnings


def compute_yearly_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "rainfall_mm"])
    yearly = (
        df.assign(year=df["date"].dt.year)
        .groupby("year", as_index=False)["rainfall_mm"]
        .sum()
    )
    return yearly


def compute_baseline_metrics(df_baseline: pd.DataFrame) -> dict:
    data = df_baseline.copy()
    data["month"] = data["date"].dt.month
    annual_totals = compute_yearly_totals(data)
    monsoon_totals = compute_yearly_totals(data[data["month"].between(7, 9)])
    monthly_climatology = (
        data.groupby("month")["rainfall_mm"].mean().reindex(range(1, 13))
    )
    return {
        "annual_mean": annual_totals["rainfall_mm"].mean(),
        "monsoon_mean": monsoon_totals["rainfall_mm"].mean(),
        "monthly_climatology": monthly_climatology,
    }


def compute_future_metrics(df_future: pd.DataFrame) -> dict:
    data = df_future.copy()
    data["month"] = data["date"].dt.month
    annual_totals = compute_yearly_totals(data)
    monsoon_totals = compute_yearly_totals(data[data["month"].between(7, 9)])
    monthly_climatology = (
        data.groupby("month")["rainfall_mm"].mean().reindex(range(1, 13))
    )
    return {
        "annual_mean": annual_totals["rainfall_mm"].mean(),
        "monsoon_mean": monsoon_totals["rainfall_mm"].mean(),
        "monthly_climatology": monthly_climatology,
    }


def summarize_changes(baseline: dict, future: dict) -> dict:
    annual_change = future["annual_mean"] - baseline["annual_mean"]
    monsoon_change = future["monsoon_mean"] - baseline["monsoon_mean"]
    annual_pct = (
        annual_change / baseline["annual_mean"] * 100
        if baseline["annual_mean"]
        else pd.NA
    )
    monsoon_pct = (
        monsoon_change / baseline["monsoon_mean"] * 100
        if baseline["monsoon_mean"]
        else pd.NA
    )
    return {
        "annual_change": annual_change,
        "annual_pct": annual_pct,
        "monsoon_change": monsoon_change,
        "monsoon_pct": monsoon_pct,
    }


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
        df, data_warnings = load_data_from_csv(uploaded_file)
    else:
        df, data_warnings = load_data_from_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Sample data not found. Run `python scripts/make_sample_data.py` first.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

for warning in data_warnings:
    st.warning(warning)

st.sidebar.header("Filters")
scenarios = st.sidebar.multiselect(
    "Scenario",
    options=sorted(df["scenario"].unique()),
    default=sorted(df["scenario"].unique()),
)
provinces = st.sidebar.multiselect(
    "Province",
    options=sorted(df["province"].unique()),
    default=sorted(df["province"].unique()),
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
    & df["province"].isin(provinces)
    & df["district_or_city"].isin(locations)
    & (df["date"] >= pd.to_datetime(start_date))
    & (df["date"] <= pd.to_datetime(end_date))
]

st.subheader("Data diagnostics")
if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

diagnostic_cols = st.columns(5)
diagnostic_cols[0].metric("Rows", f"{len(filtered):,}")
diagnostic_cols[1].metric("Cities", f"{filtered['district_or_city'].nunique():,}")
diagnostic_cols[2].metric("Provinces", f"{filtered['province'].nunique():,}")
diagnostic_cols[3].metric(
    "Min date", filtered["date"].min().strftime("%Y-%m-%d")
)
diagnostic_cols[4].metric(
    "Max date", filtered["date"].max().strftime("%Y-%m-%d")
)

scenario_label = "all-scenarios"
if len(scenarios) == 1:
    scenario_label = scenarios[0]
scenario_label = scenario_label.replace(" ", "-")
date_label = f"{pd.to_datetime(start_date):%Y%m%d}-{pd.to_datetime(end_date):%Y%m%d}"
filtered_filename = f"rainfall_filtered_{scenario_label}_{date_label}.csv"
st.download_button(
    "Download filtered data (CSV)",
    data=filtered.to_csv(index=False),
    file_name=filtered_filename,
    mime="text/csv",
)

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

st.subheader("Change vs Historical Baseline")
compare_by = st.radio("Compare by", options=["City", "Province"], horizontal=True)
future_periods = {
    "2015-2044": (pd.Timestamp("2015-01-01"), pd.Timestamp("2044-12-31")),
    "2045-2074": (pd.Timestamp("2045-01-01"), pd.Timestamp("2074-12-31")),
    "2075-2100": (pd.Timestamp("2075-01-01"), pd.Timestamp("2100-12-31")),
}
future_label = st.selectbox("Future period", options=list(future_periods.keys()))
future_start, future_end = future_periods[future_label]

analysis_source = df[df["scenario"].isin(scenarios)].copy()
if compare_by == "City":
    group_column = "district_or_city"
    selected_groups = locations
    analysis_source = analysis_source[
        analysis_source["district_or_city"].isin(selected_groups)
        & analysis_source["province"].isin(provinces)
    ]
else:
    group_column = "province"
    selected_groups = provinces
    analysis_source = analysis_source[analysis_source["province"].isin(selected_groups)]

baseline_start = pd.Timestamp("1980-01-01")
baseline_end = pd.Timestamp("2014-12-31")
required_future_scenarios = ["ssp245", "ssp585"]

summary_rows = []
climatology_rows = []
for group_name in selected_groups:
    if compare_by == "City":
        group_mask = analysis_source["district_or_city"] == group_name
    else:
        group_mask = analysis_source["province"] == group_name

    group_data = analysis_source[group_mask]
    baseline_data = group_data[
        (group_data["scenario"] == "historical")
        & (group_data["date"] >= baseline_start)
        & (group_data["date"] <= baseline_end)
    ]
    future_data = group_data[
        (group_data["scenario"].isin(required_future_scenarios))
        & (group_data["date"] >= future_start)
        & (group_data["date"] <= future_end)
    ]

    missing_future = [
        scenario
        for scenario in required_future_scenarios
        if future_data[future_data["scenario"] == scenario].empty
    ]
    if baseline_data.empty or missing_future:
        missing_parts = []
        if baseline_data.empty:
            missing_parts.append("historical baseline")
        if missing_future:
            missing_parts.append(f"future data for {', '.join(missing_future)}")
        st.warning(
            f"Skipping {group_name}: missing {' and '.join(missing_parts)} data."
        )
        continue

    baseline_metrics = compute_baseline_metrics(baseline_data)
    baseline_climatology = baseline_metrics["monthly_climatology"]
    for month, rainfall in baseline_climatology.items():
        climatology_rows.append(
            {
                "group_name": group_name,
                "scenario": "historical",
                "period": "Baseline",
                "month": month,
                "rainfall_mm": rainfall,
            }
        )

    for scenario in required_future_scenarios:
        scenario_future = future_data[future_data["scenario"] == scenario]
        future_metrics = compute_future_metrics(scenario_future)
        changes = summarize_changes(baseline_metrics, future_metrics)

        summary_rows.append(
            {
                "group_name": group_name,
                "scenario": scenario,
                "baseline_annual_mm": baseline_metrics["annual_mean"],
                "future_annual_mm": future_metrics["annual_mean"],
                "annual_change_mm": changes["annual_change"],
                "annual_change_pct": changes["annual_pct"],
                "baseline_monsoon_mm": baseline_metrics["monsoon_mean"],
                "future_monsoon_mm": future_metrics["monsoon_mean"],
                "monsoon_change_mm": changes["monsoon_change"],
                "monsoon_change_pct": changes["monsoon_pct"],
            }
        )

        future_climatology = future_metrics["monthly_climatology"]
        for month, rainfall in future_climatology.items():
            climatology_rows.append(
                {
                    "group_name": group_name,
                    "scenario": scenario,
                    "period": "Future",
                    "month": month,
                    "rainfall_mm": rainfall,
                }
            )

if summary_rows:
    summary_table = pd.DataFrame(summary_rows)
    baseline_filename = (
        "baseline_change_"
        f"{compare_by.lower()}_{future_label.replace(' ', '')}_"
        f"{scenario_label}_{date_label}.csv"
    )
    st.download_button(
        "Download baseline change table (CSV)",
        data=summary_table.to_csv(index=False),
        file_name=baseline_filename,
        mime="text/csv",
    )
    ssp585_order = (
        summary_table[summary_table["scenario"] == "SSP585"]
        .sort_values("annual_change_pct", ascending=False)["group_name"]
        .tolist()
    )
    if ssp585_order:
        remaining_groups = [
            name
            for name in summary_table["group_name"].unique()
            if name not in ssp585_order
        ]
        group_order = ssp585_order + remaining_groups
        summary_table["group_name"] = pd.Categorical(
            summary_table["group_name"], categories=group_order, ordered=True
        )
        summary_table = summary_table.sort_values(["group_name", "scenario"])

    summary_table_display = summary_table.style.format(
        {
            "baseline_annual_mm": "{:.1f}",
            "future_annual_mm": "{:.1f}",
            "annual_change_mm": "{:.1f}",
            "annual_change_pct": "{:.1f}",
            "baseline_monsoon_mm": "{:.1f}",
            "future_monsoon_mm": "{:.1f}",
            "monsoon_change_mm": "{:.1f}",
            "monsoon_change_pct": "{:.1f}",
        }
    )
    st.dataframe(summary_table_display, use_container_width=True, hide_index=True)

    annual_chart = (
        alt.Chart(summary_table)
        .mark_bar()
        .encode(
            x=alt.X("group_name:N", title="Group", sort=None),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("annual_change_pct:Q", title="Annual change (%)", stack=None),
            color=alt.Color("scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("group_name:N", title="Group"),
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("annual_change_pct:Q", title="Annual change (%)", format=".1f"),
                alt.Tooltip("annual_change_mm:Q", title="Annual change (mm)", format=".1f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(annual_chart, use_container_width=True)

    monsoon_chart = (
        alt.Chart(summary_table)
        .mark_bar()
        .encode(
            x=alt.X("group_name:N", title="Group", sort=None),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("monsoon_change_pct:Q", title="Monsoon change (%)", stack=None),
            color=alt.Color("scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("group_name:N", title="Group"),
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip(
                    "monsoon_change_pct:Q", title="Monsoon change (%)", format=".1f"
                ),
                alt.Tooltip(
                    "monsoon_change_mm:Q", title="Monsoon change (mm)", format=".1f"
                ),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(monsoon_chart, use_container_width=True)

    climatology_df = pd.DataFrame(climatology_rows)
    base_condition = alt.datum.period == "Baseline"
    climatology_chart = (
        alt.Chart(climatology_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("rainfall_mm:Q", title="Average rainfall (mm)"),
            color=alt.condition(
                base_condition, alt.value("gray"), alt.Color("scenario:N", title="Scenario")
            ),
            strokeDash=alt.condition(
                base_condition, alt.value([4, 4]), alt.value([1, 0])
            ),
            tooltip=[
                alt.Tooltip("group_name:N", title="Group"),
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("rainfall_mm:Q", title="Rainfall (mm)", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    if len(summary_table["group_name"].unique()) > 1:
        climatology_chart = climatology_chart.facet(
            row=alt.Row("group_name:N", title="Group")
        )
    st.altair_chart(climatology_chart, use_container_width=True)
else:
    st.info("No groups available for baseline comparison with the selected filters.")

summary = (
    filtered_view.groupby(["scenario", "province", "district_or_city"], as_index=False)[
        "rainfall_mm"
    ]
    .mean()
    .sort_values(["scenario", "province", "district_or_city"])
)

st.subheader("Average Monthly Rainfall (mm)")
summary_display = summary.style.format({"rainfall_mm": "{:.1f}"})
st.dataframe(summary_display, use_container_width=True, hide_index=True)

st.caption("Sample data is synthetic and for demonstration only.")
