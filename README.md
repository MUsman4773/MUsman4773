# Pakistan Rainfall Scenario Explorer

A starter Streamlit app for exploring synthetic rainfall scenarios for major Pakistani cities from 1980 through 2100.

## Project Structure

- `app.py` - Streamlit application.
- `scripts/make_sample_data.py` - Generates the sample dataset.
- `data/` - Output folder for `pakistan_rainfall_sample.csv`.

## Dataset

The generated dataset includes:

- **Historical**: 1980-01 to 2014-12 (monthly)
- **Projections**: 2015-01 to 2100-12 (monthly)
- **Scenarios**: `historical`, `ssp245`, `ssp585`
- **Locations**: Karachi (Sindh), Lahore (Punjab), Peshawar (KPK), Quetta (Balochistan), Islamabad (ICT)
- **Columns**: `date`, `province`, `district_or_city`, `scenario`, `rainfall_mm`

## Upload your own CSV

You can switch the sidebar **Data source** to **Upload my CSV** and select your own file.
The app validates the file before loading; if there is an issue it will show an error and stop.

### Required columns

- `date`
- `province`
- `district_or_city`
- `scenario`
- `rainfall_mm`

### Date formats

The `date` column must be parseable as either:

- `YYYY-MM` (monthly)
- `YYYY-MM-DD` (daily)

## Run

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py
streamlit run app.py
```
