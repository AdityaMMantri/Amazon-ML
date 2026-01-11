import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import ViTModel, ViTImageProcessor
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR=r""
CSV_PATH=r""
CHECKPOINT = r""

BATCH_SIZE = 4
MAX_LEN = 256

NUM_COLS = [
    "log_quantity",
    "is_bulk",
    "brand_present",
    "brand_price_enc"
]

PLACEHOLDER_IMAGE = Image.new("RGB", (224, 224), (128, 128, 128))

# ============================================================
# DATASET
# ============================================================
class ProductDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["sample_id"]).strip()

        text = row["catalog_content"]

        img_path = os.path.join(self.image_dir, f"{sid}.jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = PLACEHOLDER_IMAGE

        numeric = torch.from_numpy(
            row[NUM_COLS].to_numpy(dtype=np.float32)
        )

        y_log = torch.tensor(
            math.log1p(row["price"]),
            dtype=torch.float32
        )

        return text, image, numeric, y_log


def collate_fn(batch):
    texts, images, nums, y_log = zip(*batch)
    return list(texts), list(images), torch.stack(nums), torch.stack(y_log)

# ============================================================
# MODEL
# ============================================================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, t, i):
        gate = torch.sigmoid(self.fc(torch.cat([t, i], dim=1)))
        return gate * t + (1 - gate) * i


class PriceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        self.text_encoder = AutoModel.from_pretrained("microsoft/mpnet-base")

        self.image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.image_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            add_pooling_layer=False
        )

        self.fusion = GatedFusion(768)

        self.num_mlp = nn.Sequential(
            nn.Linear(len(NUM_COLS), 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(768 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode_text(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        out = self.text_encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1)

    def encode_image(self, images):
        inp = self.image_processor(images=images, return_tensors="pt").to(DEVICE)
        out = self.image_encoder(**inp).last_hidden_state
        return out[:, 0, :]

    def forward(self, texts, images, numeric):
        t = self.encode_text(texts)
        i = self.encode_image(images)
        fused = self.fusion(t, i)
        num = self.num_mlp(numeric)
        return self.head(torch.cat([fused, num], dim=1)).squeeze(1)

# ============================================================
# sMAPE
# ============================================================
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(
        2.0 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100

# ============================================================
# LOAD FULL DATASET
# ============================================================
df = pd.read_csv(CSV_PATH)

for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[NUM_COLS] = df[NUM_COLS].fillna(0.0)

# IMPORTANT:
# Fit scaler exactly once (same distribution as training)
scaler = StandardScaler()
df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS].astype(np.float32))

full_ds = ProductDataset(df, IMAGE_DIR)
full_loader = DataLoader(
    full_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ============================================================
# LOAD MODEL
# ============================================================
model = PriceModel().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ============================================================
# INFERENCE + sMAPE (ALL 75K)
# ============================================================
y_true_price = []
y_pred_price = []

with torch.no_grad():
    for texts, images, nums, y_log in tqdm(
        full_loader,
        desc="Computing sMAPE on ALL samples"
    ):
        nums = nums.to(DEVICE)

        pred_log = model(texts, images, nums)

        pred_price = torch.expm1(pred_log).cpu().numpy()
        true_price = torch.expm1(y_log).numpy()

        y_pred_price.extend(pred_price)
        y_true_price.extend(true_price)

smape_value = smape(y_true_price, y_pred_price)

print("\n================ RESULT =================")
print(f"sMAPE on ALL 75,000 samples: {smape_value:.2f}%")
print("========================================\n")
