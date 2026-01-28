import altair as alt
import pandas as pd
import pydeck as pdk
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
DEFAULTS = {
    "data_source_mode": "Use sample data",
    "uploaded_file": None,
    "scenarios": [],
    "provinces": [],
    "locations": [],
    "date_range": None,
    "monsoon_only": False,
    "compare_by": "City",
    "baseline_years": None,
    "future_years": None,
}

st.set_page_config(page_title="Pakistan Rainfall Scenarios", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.75rem;
            padding: 0.75rem;
        }
        div[data-testid="stMetric"] > div {
            gap: 0.25rem;
        }
        div[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }
        div[data-testid="stSidebar"] .stVerticalBlock > div {
            gap: 0.45rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        div[data-testid="stExpander"] > details {
            padding: 0.35rem 0.15rem;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pakistan Rainfall Scenario Explorer")
st.write(
    "Explore historical rainfall and future projections for major Pakistani cities. "
    "Use the sidebar filters to compare scenarios across time."
)

DATE_FORMATS = ["%Y-%m", "%Y-%m-%d", "%Y/%m/%d"]
MM_FORMAT = "{:.1f}"
PERCENT_FORMAT = "{:.1f}%"
CITY_COORDS = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.975),
}
PROVINCE_COLORS = [
    (33, 150, 243),
    (156, 39, 176),
    (255, 152, 0),
    (76, 175, 80),
    (244, 67, 54),
    (0, 150, 136),
]


def build_map_points(
    data: pd.DataFrame,
    scenarios_selected: list[str],
    season_label: str,
    color_by_province: bool,
) -> pd.DataFrame:
    points = data[["district_or_city", "province"]].drop_duplicates().copy()
    points["coords"] = points["district_or_city"].map(CITY_COORDS)
    points = points.dropna(subset=["coords"])
    if points.empty:
        return points
    points["lat"] = points["coords"].apply(lambda value: value[0])
    points["lon"] = points["coords"].apply(lambda value: value[1])
    points["scenario_label"] = ", ".join(scenarios_selected) or "All"
    points["season_label"] = season_label

    if color_by_province:
        province_order = sorted(points["province"].unique())
        color_map = {
            province: PROVINCE_COLORS[idx % len(PROVINCE_COLORS)]
            for idx, province in enumerate(province_order)
        }
        points["color"] = points["province"].map(color_map)
    else:
        points["color"] = [(33, 150, 243)] * len(points)
    return points


def render_color_legend(items: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not items:
        return
    legend_html = " ".join(
        [
            "<span style='display:inline-flex; align-items:center; gap:0.35rem; "
            "margin-right:0.75rem;'>"
            f"<span style='width:0.75rem; height:0.75rem; border-radius:0.25rem; "
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]});'></span>"
            f"<span>{label}</span></span>"
            for label, color in items
        ]
    )
    st.markdown(legend_html, unsafe_allow_html=True)


def section_header(title: str) -> None:
    st.markdown("")
    st.subheader(title)


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


@st.cache_data(show_spinner="Loading rainfall data...")
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


@st.cache_data(show_spinner="Applying filters...")
def apply_filters(
    df: pd.DataFrame,
    scenarios: tuple[str, ...],
    provinces: tuple[str, ...],
    locations: tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    filtered = df[
        df["scenario"].isin(scenarios)
        & df["province"].isin(provinces)
        & df["district_or_city"].isin(locations)
        & (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ].copy()
    filtered["month"] = filtered["date"].dt.month
    return filtered


def compute_yearly_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "rainfall_mm"])
    yearly = (
        df.assign(year=df["date"].dt.year)
        .groupby("year", as_index=False)["rainfall_mm"]
        .sum()
    )
    return yearly


@st.cache_data(show_spinner="Preparing overview metrics...")
def compute_overview_tables(
    filtered: pd.DataFrame,
    filtered_view: pd.DataFrame,
) -> dict:
    diagnostics = {
        "rows": len(filtered),
        "cities": filtered["district_or_city"].nunique(),
        "provinces": filtered["province"].nunique(),
        "min_date": filtered["date"].min(),
        "max_date": filtered["date"].max(),
    }

    summary_stats = {
        "total_rainfall": filtered_view["rainfall_mm"].sum(),
        "avg_rainfall": filtered_view["rainfall_mm"].mean(),
        "max_rainfall": filtered_view["rainfall_mm"].max(),
        "min_rainfall": filtered_view["rainfall_mm"].min(),
    }

    climatology = (
        filtered_view.groupby(["scenario", "month"], as_index=False)["rainfall_mm"]
        .mean()
        .sort_values(["month", "scenario"])
    )

    monsoon_data = filtered[filtered["month"].between(7, 9)]
    monsoon_summary = None
    available_scenarios = []
    if not monsoon_data.empty:
        scenario_order = ["historical", "ssp245", "ssp585"]
        available_scenarios = [
            scenario
            for scenario in scenario_order
            if scenario in monsoon_data["scenario"].unique()
        ]
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

    avg_monthly = (
        filtered_view.groupby(
            ["scenario", "province", "district_or_city"], as_index=False
        )["rainfall_mm"]
        .mean()
        .sort_values(["scenario", "province", "district_or_city"])
    )

    return {
        "diagnostics": diagnostics,
        "summary_stats": summary_stats,
        "climatology": climatology,
        "monsoon_data": monsoon_data,
        "monsoon_summary": monsoon_summary,
        "monsoon_scenarios": available_scenarios,
        "avg_monthly": avg_monthly,
    }


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


@st.cache_data(show_spinner="Computing baseline comparison...")
def compute_baseline_comparison(
    df: pd.DataFrame,
    scenarios: tuple[str, ...],
    provinces: tuple[str, ...],
    locations: tuple[str, ...],
    compare_by: str,
    baseline_range: tuple[int, int],
    future_range: tuple[int, int],
    monsoon_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    baseline_start = pd.Timestamp(f"{baseline_range[0]}-01-01")
    baseline_end = pd.Timestamp(f"{baseline_range[1]}-12-31")
    future_start = pd.Timestamp(f"{future_range[0]}-01-01")
    future_end = pd.Timestamp(f"{future_range[1]}-12-31")

    analysis_source = df[df["scenario"].isin(scenarios)].copy()
    if compare_by == "City":
        selected_groups = list(locations)
        analysis_source = analysis_source[
            analysis_source["district_or_city"].isin(selected_groups)
            & analysis_source["province"].isin(provinces)
        ]
    else:
        selected_groups = list(provinces)
        analysis_source = analysis_source[
            analysis_source["province"].isin(selected_groups)
        ]

    required_future_scenarios = ["ssp245", "ssp585"]

    summary_rows = []
    climatology_rows = []
    warnings = []
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

        if monsoon_only:
            baseline_data = baseline_data[baseline_data["date"].dt.month.between(7, 9)]
            future_data = future_data[future_data["date"].dt.month.between(7, 9)]

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
            warnings.append(
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

    summary_table = pd.DataFrame(summary_rows)
    climatology_df = pd.DataFrame(climatology_rows)

    if not summary_table.empty:
        ssp585_order = (
            summary_table[summary_table["scenario"] == "ssp585"]
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

    return summary_table, climatology_df, warnings


def clamp_year_range(
    year_min: int, year_max: int, start_year: int, end_year: int
) -> tuple[int, int]:
    clamped_start = max(year_min, start_year)
    clamped_end = min(year_max, end_year)
    if clamped_start > clamped_end:
        return year_min, year_max
    return clamped_start, clamped_end


def ensure_multiselect_state(
    key: str, options: list[str], default: list[str]
) -> None:
    if key not in st.session_state:
        st.session_state[key] = list(default)
        return

    current = list(st.session_state.get(key) or [])
    valid = [value for value in current if value in options]
    if not valid and current:
        st.session_state[key] = list(default)
    else:
        st.session_state[key] = valid


def ensure_date_range_state(
    key: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    default: tuple[pd.Timestamp, pd.Timestamp],
) -> None:
    current = st.session_state.get(key, default)
    if not isinstance(current, (list, tuple)) or len(current) != 2:
        current = default
    start_date, end_date = current

    min_ts = pd.to_datetime(min_date)
    max_ts = pd.to_datetime(max_date)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if start_ts < min_ts:
        start_ts = min_ts
    if end_ts > max_ts:
        end_ts = max_ts
    if start_ts > end_ts:
        start_ts = pd.to_datetime(default[0])
        end_ts = pd.to_datetime(default[1])

    st.session_state[key] = (start_ts.date(), end_ts.date())


def ensure_year_range_state(
    key: str,
    year_min: int,
    year_max: int,
    default: tuple[int, int],
) -> None:
    current = st.session_state.get(key, default)
    if not isinstance(current, (list, tuple)) or len(current) != 2:
        current = default
    start_year, end_year = current
    start_year = max(year_min, start_year)
    end_year = min(year_max, end_year)
    if start_year > end_year:
        start_year, end_year = default
    st.session_state[key] = (start_year, end_year)


if "data_source_mode" not in st.session_state:
    st.session_state["data_source_mode"] = DEFAULTS["data_source_mode"]
elif st.session_state["data_source_mode"] not in (
    "Use sample data",
    "Upload my CSV",
):
    st.session_state["data_source_mode"] = DEFAULTS["data_source_mode"]

reset_clicked = st.sidebar.button("Reset filters", key="reset_filters")
st.sidebar.caption("Restores defaults and reruns.")

with st.sidebar.expander("Data source", expanded=True):
    data_source = st.radio(
        "Choose a data source",
        options=("Use sample data", "Upload my CSV"),
        key="data_source_mode",
    )

    template_rows = [
        ["1980-01", "Sindh", "Karachi", "historical", 12.5],
        ["2050-07-01", "Punjab", "Lahore", "ssp245", 180.2],
    ]
    template_df = pd.DataFrame(template_rows, columns=TEMPLATE_COLUMNS)
    st.download_button(
        "Download template CSV",
        data=template_df.to_csv(index=False),
        file_name="rainfall_template.csv",
        mime="text/csv",
    )

    uploaded_file = None
    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key="uploaded_file",
        )

try:
    if data_source == "Upload my CSV":
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

scenario_options = sorted(df["scenario"].unique())
province_options = sorted(df["province"].unique())
location_options = sorted(df["district_or_city"].unique())

min_date = df["date"].min().date()
max_date = df["date"].max().date()

year_min = int(df["date"].dt.year.min())
year_max = int(df["date"].dt.year.max())

baseline_default = clamp_year_range(year_min, year_max, 1981, 2010)
future_default = clamp_year_range(year_min, year_max, 2031, 2060)

DEFAULTS.update(
    {
        "scenarios": scenario_options,
        "provinces": province_options,
        "locations": location_options,
        "date_range": (min_date, max_date),
        "monsoon_only": False,
        "compare_by": "City",
        "baseline_years": baseline_default,
        "future_years": future_default,
    }
)

if reset_clicked:
    for key in [
        "scenarios",
        "provinces",
        "locations",
        "compare_by",
        "monsoon_only",
        "baseline_years",
        "future_years",
        "data_source_mode",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

ensure_multiselect_state("scenarios", scenario_options, DEFAULTS["scenarios"])
ensure_multiselect_state("provinces", province_options, DEFAULTS["provinces"])
ensure_multiselect_state("locations", location_options, DEFAULTS["locations"])
ensure_date_range_state("date_range", min_date, max_date, DEFAULTS["date_range"])
ensure_year_range_state("baseline_years", year_min, year_max, DEFAULTS["baseline_years"])
ensure_year_range_state("future_years", year_min, year_max, DEFAULTS["future_years"])

if st.session_state.get("compare_by") not in {"City", "Province"}:
    st.session_state["compare_by"] = DEFAULTS["compare_by"]
if "monsoon_only" not in st.session_state:
    st.session_state["monsoon_only"] = DEFAULTS["monsoon_only"]

with st.sidebar.expander("Filters", expanded=True):
    scenarios = st.multiselect(
        "Scenario",
        options=scenario_options,
        key="scenarios",
    )
    st.caption("Filter rainfall data by emissions pathway.")
    provinces = st.multiselect(
        "Province",
        options=province_options,
        key="provinces",
    )
    st.caption("Limit results to specific provinces.")
    locations = st.multiselect(
        "Location",
        options=location_options,
        key="locations",
    )
    st.caption("Focus on the cities or districts you care about.")
    start_date, end_date = st.date_input(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )
    st.caption("Adjust the time window used across all charts.")
    monsoon_only = st.toggle(
        "Monsoon season only (Jul-Sep)",
        key="monsoon_only",
    )
    st.caption("Restrict analysis to monsoon months only.")
    baseline_range = st.slider(
        "Historical baseline years",
        min_value=year_min,
        max_value=year_max,
        step=1,
        key="baseline_years",
    )
    st.caption("Select the historical window used as the baseline reference.")
    future_range = st.slider(
        "Future scenario years",
        min_value=year_min,
        max_value=year_max,
        step=1,
        key="future_years",
    )
    st.caption("Set the future window for scenario comparisons.")
    compare_by = st.radio(
        "Compare by",
        options=["City", "Province"],
        horizontal=True,
        key="compare_by",
    )
    st.caption("Choose the grouping level for baseline comparisons.")

filtered = apply_filters(
    df,
    tuple(scenarios),
    tuple(provinces),
    tuple(locations),
    pd.to_datetime(start_date),
    pd.to_datetime(end_date),
)

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

scenario_label = "all-scenarios"
if len(scenarios) == 1:
    scenario_label = scenarios[0]
scenario_label = scenario_label.replace(" ", "-")
date_label = f"{pd.to_datetime(start_date):%Y%m%d}-{pd.to_datetime(end_date):%Y%m%d}"
filtered_filename = f"rainfall_filtered_{scenario_label}_{date_label}.csv"

filtered_view = filtered.copy()
if monsoon_only:
    filtered_view = filtered_view[filtered_view["month"].between(7, 9)]
    if filtered_view.empty:
        st.warning("No data for the selected filters in monsoon months.")
        st.stop()
    filtered_filename = (
        f"rainfall_filtered_{scenario_label}_{date_label}_monsoon.csv"
    )

map_tab, overview_tab, baseline_tab, data_table_tab = st.tabs(
    ["Map view", "Overview", "Baseline comparison", "Data table"]
)

with map_tab:
    section_header("Interactive map")
    season_label = "Monsoon (Jul-Sep)" if monsoon_only else "Annual"
    if "show_markers" not in st.session_state:
        st.session_state["show_markers"] = True
    if "color_by_province" not in st.session_state:
        st.session_state["color_by_province"] = compare_by == "Province"

    with st.sidebar.expander("Map tools", expanded=False):
        if st.button("Reset map", key="reset_map"):
            st.session_state.pop("map_view_state", None)
        st.toggle(
            "Show city markers",
            key="show_markers",
        )
        st.toggle(
            "Color by province",
            key="color_by_province",
            disabled=compare_by != "Province",
            help="Enable when grouping by province to color points.",
        )

    map_points = build_map_points(
        filtered,
        scenarios,
        season_label,
        st.session_state.get("color_by_province", False),
    )

    if map_points.empty:
        st.info(
            "No coordinates available for the selected cities. "
            "Add latitude/longitude data to your CSV or choose from the sample cities."
        )
    else:
        map_center = (
            map_points["lat"].mean(),
            map_points["lon"].mean(),
        )
        default_view = pdk.ViewState(
            latitude=map_center[0],
            longitude=map_center[1],
            zoom=4.8,
            pitch=20,
        )
        view_state = st.session_state.get("map_view_state", default_view)
        if not isinstance(view_state, pdk.ViewState):
            view_state = default_view
        st.session_state["map_view_state"] = view_state

        layers = []
        if st.session_state.get("show_markers", True):
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_points,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=12000,
                    pickable=True,
                    auto_highlight=True,
                )
            )

        tooltip = {
            "html": (
                "<b>{district_or_city}</b><br/>"
                "Province: {province}<br/>"
                "Scenarios: {scenario_label}<br/>"
                f"Season: {season_label}"
            ),
            "style": {
                "backgroundColor": "#111827",
                "color": "white",
                "fontSize": "0.85rem",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="mapbox://styles/mapbox/light-v10",
            ),
            use_container_width=True,
        )

        if st.session_state.get("color_by_province", False):
            province_order = sorted(map_points["province"].unique())
            legend_items = [
                (province, PROVINCE_COLORS[idx % len(PROVINCE_COLORS)])
                for idx, province in enumerate(province_order)
            ]
            st.caption("Province color key")
            render_color_legend(legend_items)
        else:
            st.caption("Markers represent selected cities.")

with overview_tab:
    overview_tables = compute_overview_tables(filtered, filtered_view)

    section_header("Data diagnostics")

    diagnostic_cols = st.columns(5)
    diagnostic_cols[0].metric("Rows", f"{overview_tables['diagnostics']['rows']:,}")
    diagnostic_cols[1].metric(
        "Cities", f"{overview_tables['diagnostics']['cities']:,}"
    )
    diagnostic_cols[2].metric(
        "Provinces", f"{overview_tables['diagnostics']['provinces']:,}"
    )
    diagnostic_cols[3].metric(
        "Min date", overview_tables["diagnostics"]["min_date"].strftime("%Y-%m-%d")
    )
    diagnostic_cols[4].metric(
        "Max date", overview_tables["diagnostics"]["max_date"].strftime("%Y-%m-%d")
    )

    section_header("Summary Statistics")
    total_rainfall = overview_tables["summary_stats"]["total_rainfall"]
    avg_rainfall = overview_tables["summary_stats"]["avg_rainfall"]
    max_rainfall = overview_tables["summary_stats"]["max_rainfall"]
    min_rainfall = overview_tables["summary_stats"]["min_rainfall"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Total rainfall (mm)", f"{total_rainfall:,.1f}")
    summary_cols[1].metric("Average rainfall (mm)", f"{avg_rainfall:,.1f}")
    summary_cols[2].metric("Maximum rainfall (mm)", f"{max_rainfall:,.1f}")
    summary_cols[3].metric("Minimum rainfall (mm)", f"{min_rainfall:,.1f}")

    section_header("Monthly Rainfall Over Time")
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

    section_header("Monthly Climatology")
    climatology = overview_tables["climatology"]
    st.download_button(
        "Download monthly climatology (CSV)",
        data=climatology.to_csv(index=False),
        file_name=f"monthly_climatology_{scenario_label}_{date_label}.csv",
        mime="text/csv",
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

    section_header("Monsoon Analysis (Jul-Sep)")
    monsoon_data = overview_tables["monsoon_data"]
    if monsoon_data.empty:
        st.info("No monsoon-season data for the selected filters.")
    else:
        monsoon_total = monsoon_data["rainfall_mm"].sum()
        monsoon_avg = monsoon_data["rainfall_mm"].mean()
        monsoon_cols = st.columns(2)
        monsoon_cols[0].metric("Monsoon total rainfall (mm)", f"{monsoon_total:,.1f}")
        monsoon_cols[1].metric(
            "Monsoon average rainfall (mm)", f"{monsoon_avg:,.1f}"
        )

        monsoon_summary = overview_tables["monsoon_summary"]
        monsoon_totals_chart = (
            alt.Chart(monsoon_summary)
            .mark_bar()
            .encode(
                x=alt.X("scenario:N", title="Scenario"),
                y=alt.Y("total_rainfall:Q", title="Total rainfall (mm)"),
                color=alt.Color("scenario:N", title="Scenario"),
                tooltip=[
                    alt.Tooltip("scenario:N", title="Scenario"),
                    alt.Tooltip(
                        "total_rainfall:Q", title="Total rainfall (mm)", format=".1f"
                    ),
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
                        "average_rainfall:Q",
                        title="Average rainfall (mm)",
                        format=".1f",
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

    section_header("Average Monthly Rainfall (mm)")
    summary = overview_tables["avg_monthly"]
    summary_display = summary.style.format({"rainfall_mm": MM_FORMAT})
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

with baseline_tab:
    section_header("Change vs Historical Baseline")

    summary_table, climatology_df, baseline_warnings = compute_baseline_comparison(
        df,
        tuple(scenarios),
        tuple(provinces),
        tuple(locations),
        compare_by,
        baseline_range,
        future_range,
        monsoon_only,
    )

    for warning in baseline_warnings:
        st.warning(warning)

    if not summary_table.empty:
        baseline_filename = (
            "baseline_change_"
            f"{compare_by.lower()}_{baseline_range[0]}-{baseline_range[1]}_"
            f"{future_range[0]}-{future_range[1]}_"
            f"{scenario_label}_{date_label}.csv"
        )
        st.download_button(
            "Download baseline comparison (CSV)",
            data=summary_table.to_csv(index=False),
            file_name=baseline_filename,
            mime="text/csv",
        )

        summary_table_display = summary_table.style.format(
            {
                "baseline_annual_mm": MM_FORMAT,
                "future_annual_mm": MM_FORMAT,
                "annual_change_mm": MM_FORMAT,
                "annual_change_pct": PERCENT_FORMAT,
                "baseline_monsoon_mm": MM_FORMAT,
                "future_monsoon_mm": MM_FORMAT,
                "monsoon_change_mm": MM_FORMAT,
                "monsoon_change_pct": PERCENT_FORMAT,
            }
        )
        st.dataframe(summary_table_display, use_container_width=True, hide_index=True)

        scenario_order = [
            scenario
            for scenario in ["ssp245", "ssp585"]
            if scenario in summary_table["scenario"].unique()
        ]
        annual_chart = (
            alt.Chart(summary_table)
            .mark_bar()
            .encode(
                x=alt.X("group_name:N", title="Group", sort=None),
                xOffset=alt.XOffset("scenario:N", sort=scenario_order or None),
                y=alt.Y("annual_change_pct:Q", title="Annual change (%)", stack=None),
                color=alt.Color("scenario:N", title="Scenario"),
                tooltip=[
                    alt.Tooltip("group_name:N", title="Group"),
                    alt.Tooltip("scenario:N", title="Scenario"),
                    alt.Tooltip(
                        "baseline_annual_mm:Q",
                        title="Baseline annual (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "future_annual_mm:Q",
                        title="Future annual (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "annual_change_mm:Q",
                        title="Annual change (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "annual_change_pct:Q",
                        title="Annual change (%)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "baseline_monsoon_mm:Q",
                        title="Baseline monsoon (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "future_monsoon_mm:Q",
                        title="Future monsoon (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "monsoon_change_mm:Q",
                        title="Monsoon change (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "monsoon_change_pct:Q",
                        title="Monsoon change (%)",
                        format=".1f",
                    ),
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
                xOffset=alt.XOffset("scenario:N", sort=scenario_order or None),
                y=alt.Y("monsoon_change_pct:Q", title="Monsoon change (%)", stack=None),
                color=alt.Color("scenario:N", title="Scenario"),
                tooltip=[
                    alt.Tooltip("group_name:N", title="Group"),
                    alt.Tooltip("scenario:N", title="Scenario"),
                    alt.Tooltip(
                        "baseline_annual_mm:Q",
                        title="Baseline annual (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "future_annual_mm:Q",
                        title="Future annual (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "annual_change_mm:Q",
                        title="Annual change (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "annual_change_pct:Q",
                        title="Annual change (%)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "baseline_monsoon_mm:Q",
                        title="Baseline monsoon (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "future_monsoon_mm:Q",
                        title="Future monsoon (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "monsoon_change_mm:Q",
                        title="Monsoon change (mm)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "monsoon_change_pct:Q",
                        title="Monsoon change (%)",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(monsoon_chart, use_container_width=True)

        base_condition = alt.datum.period == "Baseline"
        climatology_chart = (
            alt.Chart(climatology_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:O", title="Month"),
                y=alt.Y("rainfall_mm:Q", title="Average rainfall (mm)"),
                color=alt.condition(
                    base_condition,
                    alt.value("gray"),
                    alt.Color("scenario:N", title="Scenario"),
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

with data_table_tab:
    section_header("Filtered data")
    download_label = "Download filtered data (CSV)"
    filtered_download = filtered_view
    if monsoon_only:
        download_label = "Download filtered monsoon data (CSV)"
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            download_label,
            data=filtered_download.to_csv(index=False),
            file_name=filtered_filename,
            mime="text/csv",
        )
    with download_cols[1]:
        st.download_button(
            "Download filtered data (JSON)",
            data=filtered_download.to_json(orient="records"),
            file_name=filtered_filename.replace(".csv", ".json"),
            mime="application/json",
        )
    st.dataframe(filtered_download, use_container_width=True, hide_index=True)

st.caption("Sample data is synthetic and for demonstration only.")
