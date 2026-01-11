import pandas as pd
import re

df=pd.read_csv(r"")
print(df.head())

df["brand_present"]=(df["brand"].notna()&(df["brand"]!="unknown_brand")).astype(int)
print(df["brand_present"].head())

Bulk_patters=re.compile(r"\b(pack of|pack|case|box|count|ct|pcs|bags|bottles|pods|cans)\b",re.IGNORECASE)
def is_bulk(catalog_text):
    if not isinstance(catalog_text, str):
        return 0
    return int(bool(Bulk_patters.search(catalog_text)))
df["is_bulk"] = df["catalog_content"].apply(is_bulk)
print(df["is_bulk"].head())

df.to_csv(r"D:\AMAZON-ML\New_method\Dataset\test_final_features.csv",index=False)
