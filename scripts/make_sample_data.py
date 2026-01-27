import csv
import math
from datetime import date
from pathlib import Path
from random import Random


def seasonal_signal(month: int) -> float:
    peak = math.sin((month - 1) / 12 * 2 * math.pi)
    return (peak + 1) / 2


def iter_months(start: date, end: date) -> list[date]:
    months = []
    current = start
    while current <= end:
        months.append(current)
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = date(year, month, 1)
    return months


def build_dataset() -> list[dict[str, str | float]]:
    locations = [
        ("Sindh", "Karachi"),
        ("Punjab", "Lahore"),
        ("KPK", "Peshawar"),
        ("Balochistan", "Quetta"),
        ("ICT", "Islamabad"),
    ]
    historical_dates = iter_months(date(1980, 1, 1), date(2014, 12, 1))
    projection_dates = iter_months(date(2015, 1, 1), date(2100, 12, 1))

    rng = Random(42)
    rows = []

    for province, city in locations:
        base = rng.uniform(20, 60)
        seasonal_amp = rng.uniform(50, 110)
        city_bias = rng.uniform(-8, 8)
        historical_noise = rng.uniform(6, 14)

        for date_item in historical_dates:
            seasonal = seasonal_signal(date_item.month)
            rainfall = base + seasonal_amp * seasonal + city_bias
            rainfall += rng.gauss(0, historical_noise)
            rainfall = max(rainfall, 0)
            rows.append(
                {
                    "date": date_item.strftime("%Y-%m"),
                    "province": province,
                    "district_or_city": city,
                    "scenario": "historical",
                    "rainfall_mm": round(rainfall, 1),
                }
            )

        for scenario, trend in [("ssp245", 0.08), ("ssp585", 0.14)]:
            for idx, date_item in enumerate(projection_dates):
                seasonal = seasonal_signal(date_item.month)
                warming = trend * idx
                rainfall = base + seasonal_amp * seasonal + city_bias + warming
                rainfall += rng.gauss(0, historical_noise + trend * 8)
                rainfall = max(rainfall, 0)
                rows.append(
                    {
                        "date": date_item.strftime("%Y-%m"),
                        "province": province,
                        "district_or_city": city,
                        "scenario": scenario,
                        "rainfall_mm": round(rainfall, 1),
                    }
                )

    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    dataset = build_dataset()
    output_path = data_dir / "pakistan_rainfall_sample.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "province", "district_or_city", "scenario", "rainfall_mm"],
        )
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Wrote {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
