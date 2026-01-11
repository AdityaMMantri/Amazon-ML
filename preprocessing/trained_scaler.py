import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

NUM_COLS = [
    "log_quantity",
    "is_bulk",
    "brand_present",
    "brand_price_enc"
]

TRAIN_CSV = r""

# Load TRAIN data
train_df = pd.read_csv(TRAIN_CSV)

# Ensure numeric + clean
for col in NUM_COLS:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

train_df[NUM_COLS] = train_df[NUM_COLS].fillna(0.0)

# Fit scaler ONLY on TRAIN
scaler = StandardScaler()
scaler.fit(train_df[NUM_COLS].astype("float32"))

# Save scaler
SCALER_PATH = r""
joblib.dump(scaler, SCALER_PATH)

print("✔ Train scaler reconstructed and saved at:")
print(SCALER_PATH)
