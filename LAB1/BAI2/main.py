from extract import extract_all_books
from transform import transform_books
from load import load_to_csv

def main():
    raw_data = extract_all_books(max_workers=20)
    clean_df = transform_books(raw_data)
    load_to_csv(clean_df, "DS102/LAB1/BAI2/data/books_to_scrape.csv")

if __name__ == "__main__":
    main()