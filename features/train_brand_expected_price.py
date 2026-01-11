import pandas as pd

train_df= pd.read_csv(r"")
print(train_df.head())

train_df["brand"]=(train_df["brand"].fillna("unknown_brand").str.lower().str.strip())
print(train_df["brand"].head(15))

global_mean=train_df["price"].mean()

brand_stats=(train_df.groupby("brand")["price"].agg(["mean","count"]).reset_index())
brand_stats.columns=["brand","brand_agv_price","brand_count"]
print(brand_stats)

alpha = 10
brand_stats["brand_price_enc"] = (
    (brand_stats["brand_agv_price"] * brand_stats["brand_count"]
     + global_mean * alpha)
    / (brand_stats["brand_count"] + alpha)
)
train_df = train_df.merge(
    brand_stats[["brand", "brand_price_enc"]],
    on="brand",
    how="left"
)
print(train_df.head(50))
print(train_df[["brand_price_enc", "price"]].head(50))

output_path = (
    r"D:\AMAZON-ML\68e8d1d70b66d_student_resource\student_resource\dataset"
    r"\test_with_brandfinal_feature.csv"
)

train_df.to_csv(output_path, index=False)

print("Saved CSV with brand column to:")
print(output_path)
