"""Generate the synthetic logistics dataset used by this portfolio project."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
START_DATE = "2024-01-01"
END_DATE = "2026-07-31"


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    depots = {
        "Depot_A": 12500,
        "Depot_B": 9200,
        "Depot_C": 7600,
        "Depot_D": 10800,
    }

    # Simplified synthetic holiday dates for modelling practice only.
    holiday_dates = {
        "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06",
        "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
        "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05",
        "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
        "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04",
        "2026-05-25",
    }

    rows = []

    for depot, base_volume in depots.items():
        for date in dates:
            day_of_week = date.dayofweek
            month = date.month
            day_of_year = date.dayofyear

            weekly_factor = {
                0: 1.15,
                1: 1.12,
                2: 1.05,
                3: 1.03,
                4: 1.08,
                5: 0.72,
                6: 0.58,
            }[day_of_week]

            annual_seasonality = 1 + 0.06 * np.sin(
                2 * np.pi * day_of_year / 365.25
            )

            black_friday_period = month == 11 and date.day >= 20
            christmas_peak = month == 12 and date.day <= 23
            january_returns_peak = month == 1 and date.day <= 10

            peak_factor = 1.0
            if black_friday_period:
                peak_factor += 0.28
            if christmas_peak:
                peak_factor += 0.38
            if january_returns_peak:
                peak_factor += 0.12

            is_bank_holiday = date.strftime("%Y-%m-%d") in holiday_dates
            holiday_factor = 0.45 if is_bank_holiday else 1.0

            temperature_c = (
                11
                + 7 * np.sin(2 * np.pi * (day_of_year - 172) / 365.25)
                + rng.normal(0, 2.5)
            )
            rainfall_mm = max(0, rng.gamma(1.6, 1.7) - 1.0)
            weather_factor = 1 + min(rainfall_mm, 15) * 0.002

            days_since_start = (date - dates[0]).days
            trend_factor = 1 + 0.00025 * days_since_start
            noise = rng.normal(0, 0.06)

            parcel_volume = (
                base_volume
                * weekly_factor
                * annual_seasonality
                * peak_factor
                * holiday_factor
                * weather_factor
                * trend_factor
                * (1 + noise)
            )

            rows.append(
                {
                    "date": date,
                    "depot": depot,
                    "temperature_c": round(temperature_c, 1),
                    "rainfall_mm": round(rainfall_mm, 1),
                    "is_bank_holiday": int(is_bank_holiday),
                    "is_black_friday_period": int(black_friday_period),
                    "is_christmas_peak": int(christmas_peak),
                    "parcel_volume": int(round(max(parcel_volume, 0))),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dhl_style_parcel_volume.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset()
    dataset.to_csv(output_path, index=False)

    print(f"Created {output_path}")
    print(f"Rows: {len(dataset):,}")
    print(
        f"Date range: {dataset['date'].min().date()} "
        f"to {dataset['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
