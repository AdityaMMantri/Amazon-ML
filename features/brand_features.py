import re
import pandas as pd

df=pd.read_csv(r"")

# -----------------------------
# Word filters
# -----------------------------
GENERIC_WORDS = {
    "organic","original","premium","natural","fresh","gluten","glutenfree",
    "vegan","keto","whole","raw","pack","packs","ounce","ounces","oz","lb",
    "pound","value","unit","free","making","gmo","perfect","gourmet",
    "infused","all","family","size","style"
}

PRODUCT_WORDS = {
    "tea","chips","sauce","rice","candy","noodles","crackers","cookies",
    "snack","snacks","drink","beverage","syrup","vinegar","oil","seasoning",
    "powder","mix","bar","bars","soup","broth","ramen","gum","chocolate",
    "pistachios","almonds","cashews","popcorn","jelly","jam","pudding",
    "cookies","cakes","stock","broth","dressing"
}

# -----------------------------
# Brand extraction
# -----------------------------
def extract_brand(catalog_content: str) -> str:
    if not isinstance(catalog_content, str):
        return "unknown_brand"

    text = catalog_content.lower()

    # ---- Extract Item Name line ONLY
    m = re.search(r"item name:\s*([^\n\r]+)", text)
    if not m:
        return "unknown_brand"

    item_name = m.group(1)

    # cut off early if description starts
    item_name = re.split(r"[:|,;\(\)-]", item_name)[0].strip()

    tokens = item_name.split()
    brand_tokens = []

    for token in tokens:
        token = re.sub(r"[^a-z']", "", token)

        if not token:
            break
        if any(c.isdigit() for c in token):
            break
        if token in GENERIC_WORDS:
            break
        if token in PRODUCT_WORDS:
            break

        brand_tokens.append(token)

    if not brand_tokens:
        return "unknown_brand"

    if brand_tokens[0] == "generic":
        return "unknown_brand"

    return " ".join(brand_tokens)

# -----------------------------
# Apply & Save
# -----------------------------
df["brand"] = df["catalog_content"].apply(extract_brand)

output_path = (
    r"D:\AMAZON-ML\68e8d1d70b66d_student_resource\student_resource\dataset"
    r"\test_with_brand.csv"
)

df.to_csv(output_path, index=False)

print("Saved CSV with brand column to:")
print(output_path)
