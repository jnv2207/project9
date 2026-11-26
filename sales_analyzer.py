"""
Pandas Analyzer & Data Visualization
------------------------------------
SalesDataAnalyzer encapsulates loading, cleaning, analysing and
visualising a sales CSV using Pandas, NumPy, Matplotlib and Seaborn.
Run:  python sales_analyzer.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ----------------------------------------------------------------------
# Core class
# ----------------------------------------------------------------------
class SalesDataAnalyzer:
    """Encapsulates all sales data analysis and visualisation features."""

    def __init__(self, file_path=None):
        self.data = None          # Pandas DataFrame holding the sales data
        self.last_fig = None      # last figure created (used for saving)
        if file_path:
            self.load_data(file_path)

    def __del__(self):
        # Cleanup: close any open matplotlib figures
        try:
            plt.close("all")
        except Exception:
            pass

    # ---------------- helpers ----------------
    def _check_loaded(self):
        if self.data is None:
            print("No dataset loaded. Please use option 1 first.")
            return False
        return True

    def _numeric_cols(self):
        return self.data.select_dtypes(include="number").columns.tolist()

    def _ask_column(self, prompt, numeric=False):
        cols = self._numeric_cols() if numeric else self.data.columns.tolist()
        print(f"Available columns: {', '.join(cols)}")
        name = input(prompt).strip()
        if name not in self.data.columns:
            print(f"Column '{name}' not found.")
            return None
        return name

    # ---------------- loading / exploring ----------------
    def load_data(self, file_path):
        """Load data from a CSV file (with exception handling)."""
        try:
            self.data = pd.read_csv(file_path)
            if "Date" in self.data.columns:
                self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
            print("Dataset loaded successfully!")
        except FileNotFoundError:
            print(f"Error: file '{file_path}' not found.")
        except pd.errors.EmptyDataError:
            print("Error: the file is empty.")
        except Exception as e:
            print(f"Error while loading file: {e}")

    def explore_data(self, choice):
        if not self._check_loaded():
            return
        if choice == 1:
            print(self.data.head())
        elif choice == 2:
            print(self.data.tail())
        elif choice == 3:
            print(list(self.data.columns))
        elif choice == 4:
            print(self.data.dtypes)
        elif choice == 5:
            self.data.info()
        else:
            print("Invalid choice.")

    # ---------------- cleaning ----------------
    def show_missing(self):
        missing = self.data[self.data.isnull().any(axis=1)]
        if missing.empty:
            print("\nNo missing values found in the dataset!")
        else:
            print(f"\nRows with missing values ({len(missing)}):")
            print(missing)
            print("\nMissing per column:\n", self.data.isnull().sum())

    def clean_data(self, choice):
        """Handle missing values / cleaning."""
        if not self._check_loaded():
            return
        if choice == 1:
            self.show_missing()
        elif choice == 2:
            nums = self._numeric_cols()
            self.data[nums] = self.data[nums].fillna(self.data[nums].mean())
            print("Missing numeric values filled with column mean.")
        elif choice == 3:
            before = len(self.data)
            self.data = self.data.dropna().reset_index(drop=True)
            print(f"Dropped {before - len(self.data)} rows with missing values.")
        elif choice == 4:
            value = input("Enter replacement value: ").strip()
            for col in self.data.columns[self.data.isnull().any()]:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    try:
                        self.data[col] = self.data[col].fillna(float(value))
                    except ValueError:
                        print(f"'{value}' is not numeric; skipped column {col}.")
                else:
                    self.data[col] = self.data[col].fillna(value)
            print("Missing values replaced.")
        else:
            print("Invalid choice.")

    def convert_types(self):
        """Basic data type conversions."""
        if not self._check_loaded():
            return
        if "Year" in self.data.columns:
            self.data["Year"] = pd.to_numeric(self.data["Year"], errors="coerce").astype("Int64")
        print("Data types after conversion:\n", self.data.dtypes)

    # ---------------- A. NumPy arrays ----------------
    def numpy_demo(self):
        """Convert columns to NumPy arrays; show indexing and slicing."""
        if not self._check_loaded() or "Sales" not in self.data.columns:
            return
        arr = self.data["Sales"].dropna().to_numpy()
        print("Sales array (first 10):", arr[:10])
        print("Shape:", arr.shape, "| dtype:", arr.dtype)
        print("First element  arr[0]   :", arr[0])
        print("Last element   arr[-1]  :", arr[-1])
        print("Slice          arr[2:7] :", arr[2:7])
        print("Every 5th      arr[::5] :", arr[::5][:10])
        print("Boolean mask   arr > mean (count):", int((arr > arr.mean()).sum()))

    # ---------------- B. math / combine / split ----------------
    def mathematical_operations(self):
        if not self._check_loaded():
            return
        d = self.data
        if {"Sales", "Profit"}.issubset(d.columns):
            d["Cost"] = d["Sales"] - d["Profit"]
            d["ProfitMargin%"] = (d["Profit"] / d["Sales"] * 100).round(2)
        if {"Sales", "Quantity"}.issubset(d.columns):
            d["UnitPrice"] = (d["Sales"] / d["Quantity"]).round(2)
        if "Sales" in d.columns:
            d["Sales_Sqrt"] = np.sqrt(d["Sales"])
            d["Sales_Log"] = np.log1p(d["Sales"])
        print("Added Cost, ProfitMargin%, UnitPrice, Sales_Sqrt, Sales_Log columns.")
        print(d.head())

    def combine_data(self, other_dataframe, how="concat"):
        """Combine the current DataFrame with another using concat/merge/join."""
        if not self._check_loaded():
            return
        if how == "concat":
            self.data = pd.concat([self.data, other_dataframe], ignore_index=True)
        elif how == "merge":
            self.data = self.data.merge(other_dataframe, on="Product", how="left")
        elif how == "join":
            self.data = self.data.join(other_dataframe.set_index("Product"), on="Product")
        print(f"Combined using {how}. New shape: {self.data.shape}")

    def split_data(self, column):
        """Split DataFrame into several DataFrames by a column's values."""
        if not self._check_loaded():
            return {}
        parts = {k: g.reset_index(drop=True) for k, g in self.data.groupby(column)}
        for key, part in parts.items():
            print(f"{column} = {key}: {len(part)} rows")
        return parts

    # ---------------- C. search / sort / filter / aggregate ----------------
    def search_sort_filter(self, action):
        if not self._check_loaded():
            return
        d = self.data
        if action == 1:  # search
            col = self._ask_column("Column to search in: ")
            if col:
                term = input("Search value: ").strip()
                res = d[d[col].astype(str).str.contains(term, case=False, na=False)]
                print(res if not res.empty else "No matching records found.")
        elif action == 2:  # sort
            col = self._ask_column("Column to sort by: ")
            if col:
                asc = input("Ascending? (y/n): ").strip().lower() == "y"
                print(d.sort_values(col, ascending=asc).head(15))
        elif action == 3:  # filter
            col = self._ask_column("Column to filter on: ")
            if col:
                val = input("Keep rows equal to (or '>number' / '<number'): ").strip()
                if val[:1] in "<>" and val[1:].replace(".", "", 1).isdigit():
                    num = float(val[1:])
                    res = d[d[col] > num] if val[0] == ">" else d[d[col] < num]
                else:
                    res = d[d[col].astype(str).str.lower() == val.lower()]
                print(res.head(20) if not res.empty else "No rows match the filter.")
        else:
            print("Invalid choice.")

    def aggregate_functions(self):
        if not self._check_loaded():
            return
        nums = self._numeric_cols()
        print("Sum:\n", self.data[nums].sum().round(2))
        print("\nMean:\n", self.data[nums].mean().round(2))
        print("\nCount:\n", self.data[nums].count())
        if {"Region", "Sales"}.issubset(self.data.columns):
            print("\nSales by Region (sum, mean, count):")
            print(self.data.groupby("Region")["Sales"].agg(["sum", "mean", "count"]).round(2))

    # ---------------- D. statistics ----------------
    def statistical_analysis(self, choice):
        if not self._check_loaded():
            return
        nums = self.data[self._numeric_cols()]
        if choice == 1:
            print(nums.describe().round(2))
        elif choice == 2:
            print("Standard deviation:\n", nums.std().round(2))
        elif choice == 3:
            print("Variance:\n", nums.var().round(2))
        elif choice == 4:
            print("Percentiles (25th, 50th, 75th, 90th):")
            print(nums.quantile([0.25, 0.5, 0.75, 0.9]).round(2))
        else:
            print("Invalid choice.")

    # ---------------- advanced operations ----------------
    def create_pivot_table(self):
        if not self._check_loaded():
            return
        if {"Region", "Product", "Sales"}.issubset(self.data.columns):
            pivot = pd.pivot_table(
                self.data, values="Sales", index="Region", columns="Product",
                aggfunc="sum", fill_value=0, margins=True,
            ).round(2)
            print(pivot)
            return pivot
        print("Required columns (Region, Product, Sales) not found.")

    def advanced_operations(self):
        """Re-index, relabel, groupby() + transform()."""
        if not self._check_loaded():
            return
        d = self.data
        # re-index + alter labels
        if "SalesID" in d.columns:
            reindexed = d.set_index("SalesID").rename(columns={"Sales": "Sales_Amount"})
            print("Re-indexed on SalesID and renamed 'Sales' -> 'Sales_Amount':")
            print(reindexed.head())
        # groupby + transform
        if {"Region", "Sales"}.issubset(d.columns):
            d["Region_AvgSales"] = d.groupby("Region")["Sales"].transform("mean").round(2)
            d["Sales_vs_RegionAvg"] = (d["Sales"] - d["Region_AvgSales"]).round(2)
            print("\nAdded Region_AvgSales and Sales_vs_RegionAvg using groupby().transform():")
            print(d[["Region", "Sales", "Region_AvgSales", "Sales_vs_RegionAvg"]].head())

    # ---------------- E. Matplotlib ----------------
    def visualize_data(self, choice):
        if not self._check_loaded():
            return
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            if choice == 1:  # bar
                grp = self.data.groupby("Region")["Sales"].sum()
                ax.bar(grp.index, grp.values, color="steelblue", label="Total Sales")
                ax.set(title="Total Sales by Region", xlabel="Region", ylabel="Sales")
            elif choice == 2:  # line
                m = self.data.dropna(subset=["Date"]).set_index("Date")["Sales"].resample("ME").sum()
                ax.plot(m.index, m.values, marker="o", label="Monthly Sales")
                ax.set(title="Monthly Sales Trend", xlabel="Month", ylabel="Sales")
            elif choice == 3:  # scatter
                x = self._ask_column("Enter x-axis column name: ", numeric=True)
                y = self._ask_column("Enter y-axis column name: ", numeric=True) if x else None
                if not (x and y):
                    plt.close(fig)
                    return
                print("Generating scatter plot...")
                ax.scatter(self.data[x], self.data[y], alpha=0.6, label=f"{y} vs {x}")
                ax.set(title=f"{y} vs {x}", xlabel=x, ylabel=y)
            elif choice == 4:  # pie
                grp = self.data.groupby("Product")["Sales"].sum()
                ax.pie(grp.values, labels=grp.index, autopct="%1.1f%%", startangle=90)
                ax.set_title("Sales Share by Product")
            elif choice == 5:  # histogram
                ax.hist(self.data["Sales"].dropna(), bins=20, color="seagreen", edgecolor="black", label="Sales")
                ax.set(title="Sales Distribution", xlabel="Sales", ylabel="Frequency")
            elif choice == 6:  # stack plot
                pv = self.data.pivot_table(values="Sales", index="Year", columns="Product", aggfunc="sum", fill_value=0)
                ax.stackplot(pv.index.astype(int), pv.T.values, labels=pv.columns)
                ax.set(title="Yearly Sales by Product (Stack Plot)", xlabel="Year", ylabel="Sales")
                ax.set_xticks(pv.index.astype(int))
            elif choice == 7:  # subplots
                plt.close(fig)
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                reg = self.data.groupby("Region")["Sales"].sum()
                prod = self.data.groupby("Product")["Sales"].sum()
                axes[0, 0].bar(reg.index, reg.values, label="Sales"); axes[0, 0].set_title("Sales by Region")
                axes[0, 1].pie(prod.values, labels=prod.index, autopct="%1.0f%%"); axes[0, 1].set_title("Product Share")
                axes[1, 0].hist(self.data["Sales"].dropna(), bins=15, color="orange", label="Sales"); axes[1, 0].set_title("Sales Histogram")
                axes[1, 1].scatter(self.data["Sales"], self.data["Profit"], alpha=0.5, label="Profit"); axes[1, 1].set_title("Sales vs Profit")
                for a in (axes[0, 0], axes[1, 0], axes[1, 1]):
                    a.legend()
                fig.suptitle("Sales Dashboard")
                fig.tight_layout()
                self.last_fig = fig
                print("Subplots displayed successfully!")
                plt.show()
                return
            elif choice == 8:  # seaborn heatmap
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(self.data[self._numeric_cols()].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                ax.set_title("Correlation Heatmap")
            elif choice == 9:  # seaborn box plot
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.boxplot(data=self.data, x="Region", y="Profit", hue="Region", palette="Set2", legend=False, ax=ax)
                ax.set_title("Profit Distribution by Region")
            else:
                print("Invalid choice.")
                plt.close(fig)
                return

            if choice in (1, 2, 3, 5, 6):
                ax.legend()
            fig.tight_layout()
            self.last_fig = fig
            if choice != 3:
                print("Plot displayed successfully!")
            else:
                print("Scatter plot displayed successfully!")
            plt.show()
        except KeyError as e:
            plt.close(fig)
            print(f"Required column missing: {e}")
        except Exception as e:
            plt.close(fig)
            print(f"Could not create plot: {e}")

    def save_visualization(self, filename):
        """Save the most recent plot as PNG/JPEG."""
        if self.last_fig is None:
            print("No plot to save. Create a visualization first (option 6).")
            return
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename += ".png"
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", filename)
        self.last_fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved as {path} successfully!")

    def export_results(self, filename="analysis_results.csv"):
        if not self._check_loaded():
            return
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", filename)
        self.data.to_csv(path, index=False)
        print(f"Results exported to {path}")

    def summary(self):
        """Conclusion: summarise key findings."""
        if not self._check_loaded():
            return
        d = self.data
        print("\n=== Summary of Findings ===")
        print(f"Records: {len(d)} | Columns: {d.shape[1]}")
        if "Sales" in d.columns:
            print(f"Total sales: {d['Sales'].sum():,.2f} | Average sale: {d['Sales'].mean():,.2f}")
        if {"Region", "Sales"}.issubset(d.columns):
            print("Top region:", d.groupby("Region")["Sales"].sum().idxmax())
        if {"Product", "Sales"}.issubset(d.columns):
            print("Top product:", d.groupby("Product")["Sales"].sum().idxmax())


# ----------------------------------------------------------------------
# Inheritance + operator overloading (optional requirement)
# ----------------------------------------------------------------------
class AdvancedSalesAnalyzer(SalesDataAnalyzer):
    """Extends SalesDataAnalyzer with extra behaviour."""

    def __init__(self, file_path=None):
        super().__init__(file_path)          # call parent constructor

    def explore_data(self, choice):          # method overriding
        print("[Advanced Analyzer]")
        super().explore_data(choice)

    def __add__(self, other):                # operator overloading: a + b
        combined = AdvancedSalesAnalyzer()
        combined.data = pd.concat([self.data, other.data], ignore_index=True)
        return combined

    def __len__(self):
        return 0 if self.data is None else len(self.data)


# ----------------------------------------------------------------------
# Menu-driven UI
# ----------------------------------------------------------------------
def read_int(prompt):
    try:
        return int(input(prompt).strip())
    except ValueError:
        return -1


def main_menu():
    print("\n========== Data Analysis & Visualization Program ==========")
    print("Please select an option:")
    for i, item in enumerate([
        "Load Dataset", "Explore Data", "Perform DataFrame Operations",
        "Handle Missing Data", "Generate Descriptive Statistics",
        "Data Visualization", "Save Visualization", "Exit"], 1):
        print(f"{i}. {item}")
    print("=" * 59)


def dataframe_operations(app):
    print("\n== DataFrame Operations ==")
    for i, t in enumerate([
        "NumPy arrays: indexing & slicing", "Mathematical operations",
        "Combine DataFrames (concat/merge/join)", "Split data by Region/Product",
        "Search / Sort / Filter", "Aggregate functions (sum, mean, count)",
        "Pivot table", "Re-index, relabel, groupby + transform",
        "Convert data types", "Summary & export results"], 1):
        print(f"{i}. {t}")
    c = read_int("Enter your choice: ")
    if c == 1:
        app.numpy_demo()
    elif c == 2:
        app.mathematical_operations()
    elif c == 3 and app._check_loaded():
        how = input("Method (concat/merge/join): ").strip().lower()
        if how == "concat":
            app.combine_data(app.data.copy(), "concat")   # demo: combine with a copy
        elif how in ("merge", "join"):
            info = pd.DataFrame({
                "Product": app.data["Product"].dropna().unique(),
                "Category": "Electronics",
            })
            app.combine_data(info, how)
        else:
            print("Invalid method.")
    elif c == 4 and app._check_loaded():
        col = input("Split by (Region/Product): ").strip()
        if col in app.data.columns:
            app.split_data(col)
        else:
            print("Invalid column.")
    elif c == 5:
        print("1. Search  2. Sort  3. Filter")
        app.search_sort_filter(read_int("Enter your choice: "))
    elif c == 6:
        app.aggregate_functions()
    elif c == 7:
        app.create_pivot_table()
    elif c == 8:
        app.advanced_operations()
    elif c == 9:
        app.convert_types()
    elif c == 10:
        app.summary()
        if input("Export data to CSV? (y/n): ").strip().lower() == "y":
            app.export_results()
    else:
        if c != 3 and c != 4:
            print("Invalid choice.")


def main():
    app = AdvancedSalesAnalyzer()
    while True:
        main_menu()
        choice = read_int("\nEnter your choice: ")

        if choice == 1:
            print("\n== Load Dataset ==")
            path = input("Enter the path of the dataset (CSV file): ").strip()
            app.load_data(path)
        elif choice == 2:
            print("\n== Explore Data ==")
            print("1. Display the first 5 rows\n2. Display the last 5 rows\n"
                  "3. Display column names\n4. Display data types\n5. Display basic info")
            app.explore_data(read_int("Enter your choice: "))
        elif choice == 3:
            dataframe_operations(app)
        elif choice == 4:
            print("\n== Handle Missing Data ==")
            print("1. Display rows with missing values\n2. Fill missing values with mean\n"
                  "3. Drop rows with missing values\n4. Replace missing values with a specific value")
            app.clean_data(read_int("Enter your choice: "))
        elif choice == 5:
            print("\n== Descriptive Statistics ==")
            print("1. describe()\n2. Standard deviation\n3. Variance\n4. Percentiles (quantile)")
            app.statistical_analysis(read_int("Enter your choice: "))
        elif choice == 6:
            print("\n== Data Visualization ==")
            print("1. Bar Plot\n2. Line Plot\n3. Scatter Plot\n4. Pie Chart\n5. Histogram\n"
                  "6. Stack Plot\n7. Subplots (dashboard)\n8. Heatmap (Seaborn)\n9. Box Plot (Seaborn)")
            app.visualize_data(read_int("Enter your choice: "))
        elif choice == 7:
            print("\n== Save Visualization ==")
            name = input("Enter file name to save the plot (e.g., scatter_plot.png): ").strip()
            app.save_visualization(name)
        elif choice == 8:
            print("\nExiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main()1