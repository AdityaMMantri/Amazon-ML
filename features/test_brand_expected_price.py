import pandas as pd
#---------------------------------------------------------TRAIN DATA--------------------------------------------------------------
train_df = pd.read_csv(r"")
brand_encodings = train_df[["brand", "brand_price_enc"]].drop_duplicates()
brand_encoding_dict = dict(zip(brand_encodings["brand"], brand_encodings["brand_price_enc"]))
global_mean = train_df["price"].mean()

#---------------------------------------------------------TEST DATA--------------------------------------------------------------
test_df = pd.read_csv(r"")
test_df["brand"] = test_df["brand"].fillna("unknown_brand").str.lower().str.strip()

# Map encodings from training
test_df["brand_price_enc"] = test_df["brand"].map(brand_encoding_dict)

# For unseen brands, use global mean
test_df["brand_price_enc"] = test_df["brand_price_enc"].fillna(global_mean)

# Save
output_path = (
    r"D:\AMAZON-ML\68e8d1d70b66d_student_resource\student_resource\dataset\test_with_brand_encoding.csv"
)

test_df.to_csv(output_path, index=False)

print(f"Missing values in brand_price_enc: {test_df['brand_price_enc'].isnull().sum()}")
print("Test set with encoding saved!")
