import re
import numpy as np
import pandas as pd

df=pd.read_csv(r"")


def extract_value_unit(text):
    if not isinstance(text, str):
        return np.nan, ""

    v = re.search(r"value:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    u = re.search(r"unit:\s*([a-zA-Z \.\(\)]+)", text, re.IGNORECASE)

    value = float(v.group(1)) if v else np.nan
    unit = u.group(1).lower() if u else ""

    return value, unit


df[["Value", "Unit"]] = df["catalog_content"].apply(
    lambda x: pd.Series(extract_value_unit(x))
)

WEIGHT_UNITS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "pound": 453.592
}

VOLUME_UNITS = {
    "ml": 1.0,
    "milliliter": 1.0,
    "liter": 1000.0,
    "litre": 1000.0,
    "fl oz": 29.5735,
    "fluid ounce": 29.5735,
    "gallon": 3785.41
}

def compute_log_quantity(row):
    value = row["Value"]
    unit = row["Unit"]

    if pd.isna(value) or not unit:
        return 0.0

    # normalize unit text
    unit_clean = re.sub(r"[^a-z ]", "", unit)

    # ---- weight
    for u in WEIGHT_UNITS:
        if u in unit_clean:
            grams = value * WEIGHT_UNITS[u]
            return np.log1p(grams)

    # ---- volume
    for u in VOLUME_UNITS:
        if u in unit_clean:
            ml = value * VOLUME_UNITS[u]
            return np.log1p(ml)

    # ---- count (robust)
    if "count" in unit_clean:
        return np.log1p(value)

    return 0.0


df["log_quantity"] = df.apply(compute_log_quantity, axis=1)

df.to_csv(r"D:\AMAZON-ML\68e8d1d70b66d_student_resource\student_resource\dataset\test_with_log_quantity_final.csv",index=False)
print(df[["Value", "Unit", "log_quantity"]].head(10))
