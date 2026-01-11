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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR=r""
CSV_PATH=r""
BATCH_SIZE=4
EPOCHS=15
LR_MLP=1e-3
LR_ENCODER=1e-5
MAX_LEN=256
NUM_COLS = [
    "log_quantity",
    "is_bulk",
    "brand_present",
    "brand_price_enc"
]

# ============================================================
# PLACEHOLDER IMAGE
# ============================================================
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

# ============================================================
# COLLATE FUNCTION
# ============================================================
def collate_fn(batch):
    texts, images, nums, y_log = zip(*batch)
    nums = torch.stack(nums)
    y_log = torch.stack(y_log)
    return list(texts), list(images), nums, y_log

# ============================================================
# GATED FUSION
# ============================================================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, t, i):
        gate = torch.sigmoid(self.fc(torch.cat([t, i], dim=1)))
        return gate * t + (1 - gate) * i

# ============================================================
# MODEL
# ============================================================
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
        inp = self.image_processor(
            images=images,
            return_tensors="pt"
        ).to(DEVICE)

        out = self.image_encoder(**inp).last_hidden_state
        return out[:, 0, :]  # CLS

    def forward(self, texts, images, numeric):
        t = self.encode_text(texts)
        i = self.encode_image(images)
        fused = self.fusion(t, i)
        num = self.num_mlp(numeric)
        return self.head(torch.cat([fused, num], dim=1)).squeeze(1)

# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    losses = []

    for texts, images, nums, y_log in tqdm(
        loader,
        desc=f"Training Epoch {epoch+1}",
        leave=True
    ):
        nums = nums.to(DEVICE)
        y_log = y_log.to(DEVICE)

        optimizer.zero_grad()
        pred = model(texts, images, nums)
        loss = nn.MSELoss()(pred, y_log)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

def evaluate(model, loader):
    model.eval()

    log_preds, log_true = [], []
    price_preds, price_true = [], []

    print(">>> Starting evaluation")

    with torch.no_grad():
        for texts, images, nums, y_log in tqdm(
            loader,
            desc="Evaluating",
            leave=True
        ):
            nums = nums.to(DEVICE)
            pred_log = model(texts, images, nums)

            log_preds.extend(pred_log.cpu().numpy())
            log_true.extend(y_log.numpy())

            price_preds.extend(np.expm1(pred_log.cpu().numpy()))
            price_true.extend(np.expm1(y_log.numpy()))

    print(">>> Evaluation finished")

    rmse_log = np.sqrt(mean_squared_error(log_true, log_preds))
    rmse_price = np.sqrt(mean_squared_error(price_true, price_preds))

    return {
        "rmse_log": rmse_log,
        "mae_log": mean_absolute_error(log_true, log_preds),
        "r2_log": r2_score(log_true, log_preds),
        "rmse_price": rmse_price,
        "mae_price": mean_absolute_error(price_true, price_preds),
    }

# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[NUM_COLS] = df[NUM_COLS].fillna(0.0)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

scaler = StandardScaler()
train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS].astype(np.float32))
val_df[NUM_COLS] = scaler.transform(val_df[NUM_COLS].astype(np.float32))

train_ds = ProductDataset(train_df, IMAGE_DIR)
val_ds = ProductDataset(val_df, IMAGE_DIR)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# ============================================================
# MODEL SETUP
# ============================================================
model = PriceModel().to(DEVICE)

for p in model.text_encoder.parameters():
    p.requires_grad = False
for p in model.image_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_MLP
)

os.makedirs("checkpoints", exist_ok=True)

# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(EPOCHS):

    print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")

    if epoch == 12:
        print(">>> Unfreezing last 2 layers of encoders")

        for layer in model.text_encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        for layer in model.image_encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": LR_MLP},
                {"params": model.num_mlp.parameters(), "lr": LR_MLP},
                {"params": model.fusion.parameters(), "lr": LR_MLP},
                {"params": model.text_encoder.encoder.layer[-2:].parameters(), "lr": LR_ENCODER},
                {"params": model.image_encoder.encoder.layer[-2:].parameters(), "lr": LR_ENCODER},
            ]
        )

    train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    metrics = evaluate(model, val_loader)

    print(f"Train MSE (log): {train_loss:.4f}")
    print(
        f"Val RMSE(log): {metrics['rmse_log']:.4f} | "
        f"MAE(log): {metrics['mae_log']:.4f} | "
        f"R2(log): {metrics['r2_log']:.4f}"
    )
    print(
        f"Val RMSE(price): {metrics['rmse_price']:.2f} | "
        f"MAE(price): {metrics['mae_price']:.2f}"
    )

    torch.save(
        model.state_dict(),
        f"checkpoints/model_epoch_{epoch+1}.pt"
    )

    print(f">>> Saved checkpoint model_epoch_{epoch+1}.pt")
