import re
import pandas as pd

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

def clean_price(price_str):
    if not price_str:
        return 0.0
    cleaned = re.sub(r"[^0-9.]", "", str(price_str))
    return float(cleaned) if cleaned else 0.0

def extract_stock_count(avail_str):
    if not avail_str:
        return 0
    match = re.search(r"\((\d+)\s+available\)", str(avail_str))
    return int(match.group(1)) if match else 0

def transform_books(raw_books):
    df = pd.DataFrame(raw_books)

    df["rating"] = df["rating_raw"].map(RATING_MAP).fillna(0).astype(int)
    df["price_excl_tax"] = df["price_excl_tax_raw"].apply(clean_price)
    df["price_incl_tax"] = df["price_incl_tax_raw"].apply(clean_price)
    df["tax"] = df["tax_raw"].apply(clean_price)
    df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False, na=False)
    df["stock_quantity"] = df["availability_raw"].apply(extract_stock_count)
    df["num_reviews"] = pd.to_numeric(df["num_reviews_raw"], errors="coerce").fillna(0).astype(int)

    columns_order = [
        "upc",
        "title",
        "category",
        "product_type",
        "rating",
        "price_excl_tax",
        "price_incl_tax",
        "tax",
        "in_stock",
        "stock_quantity",
        "num_reviews",
        "description",
        "image_url",
        "url"
    ]

    return df[columns_order]