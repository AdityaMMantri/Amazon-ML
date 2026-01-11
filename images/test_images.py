import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


CSV_PATH=r""
IMAGE_DIR="test_images"
FAILED_LOG="test_failed_downloads.txt"

MAX_WORKERS=6
REQUEST_DELAY=0.4
TIMEOUT=10
MAX_RETRIES=1

os.makedirs(IMAGE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.amazon.com/",
    "Connection": "keep-alive",
}


df = pd.read_csv(CSV_PATH)
df = df[["sample_id", "image_link"]]

print(f"Total samples: {len(df)}")


def download_one(row):
    sample_id, url = row
    filepath = os.path.join(IMAGE_DIR, f"{sample_id}.jpg")

    if os.path.exists(filepath):
        return None  # skipped

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(filepath, format="JPEG", quality=95)

        time.sleep(REQUEST_DELAY)
        return None

    except Exception as e:
        return (sample_id, str(e))

# =========================
# PARALLEL EXECUTION
# =========================
failed = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(download_one, row)
        for row in df.itertuples(index=False)
    ]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        result = future.result()
        if result is not None:
            failed.append(result)

if failed:
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        for sid, reason in failed:
            f.write(f"{sid},{reason}\n")

    print(f"\n❌ Failed downloads: {len(failed)}")
    print(f"📄 Logged in {FAILED_LOG}")
else:
    print("\n🎉 All images downloaded successfully")
