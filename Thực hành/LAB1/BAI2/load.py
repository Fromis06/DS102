def load_to_csv(df, output_path="DS102/LAB1/BAI2/data/books_to_scrape.csv"):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")