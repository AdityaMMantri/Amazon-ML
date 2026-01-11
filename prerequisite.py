import pandas as pd
import torch
df=pd.read_csv(r"D:\AMAZON-ML\New_method\Dataset\train_final_features.csv")
print(df.describe())
print(df.info())

NUM_COLS = [
    "log_quantity",
    "is_bulk",
    "brand_present",
    "brand_price_enc"
]

print(df[NUM_COLS].dtypes)

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("GPU Device Count:", torch.cuda.device_count())
    print("Current GPU Device:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("❌ CUDA is NOT available. Using CPU.")
print(torch.version.cuda)
