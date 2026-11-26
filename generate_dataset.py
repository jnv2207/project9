"""Creates a synthetic sales dataset (data/sales_data.csv)."""
import os
import numpy as np
import pandas as pd


def generate(rows=300, seed=42, path="data/sales_data.csv"):
    rng = np.random.default_rng(seed)
    products = ["Laptop", "Phone", "Tablet", "Headphones", "Monitor"]
    regions = ["North", "South", "East", "West", "Central"]

    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 3 * 365, rows), unit="D"
    )
    df = pd.DataFrame({
        "SalesID": np.arange(101, 101 + rows),
        "Date": dates.strftime("%Y-%m-%d"),
        "Product": rng.choice(products, rows),
        "Region": rng.choice(regions, rows),
        "Quantity": rng.integers(1, 20, rows),
    })
    price = df["Product"].map(
        {"Laptop": 900, "Phone": 600, "Tablet": 400, "Headphones": 120, "Monitor": 250}
    )
    df["Sales"] = (df["Quantity"] * price * rng.uniform(0.9, 1.1, rows)).round(2)
    df["Profit"] = (df["Sales"] * rng.uniform(0.08, 0.30, rows)).round(2)
    df["Year"] = pd.to_datetime(df["Date"]).dt.year

    # a few missing values so the cleaning feature has something to do
    for col in ("Sales", "Profit", "Region"):
        df.loc[rng.choice(rows, 5, replace=False), col] = np.nan

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path} ({rows} rows)")


if __name__ == "__main__":
    generate()