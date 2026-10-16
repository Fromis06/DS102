import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://books.toscrape.com/"

def get_all_book_urls():
    book_urls = []
    page_url = urljoin(BASE_URL, "catalogue/page-1.html")
    session = requests.Session()

    while page_url:
        response = session.get(page_url, timeout=15)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        for h3 in soup.select("article.product_pod h3 a"):
            rel_url = h3["href"]
            book_urls.append(urljoin(page_url, rel_url))

        next_btn = soup.select_one("li.next a")
        if next_btn:
            page_url = urljoin(page_url, next_btn["href"])
        else:
            page_url = None

    return book_urls

def fetch_book_detail(url):
    response = requests.get(url, timeout=15)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.select_one("div.product_main h1").text.strip()
    
    rating_p = soup.select_one("p.star-rating")
    rating_classes = rating_p.get("class", []) if rating_p else []
    rating = next((c for c in rating_classes if c != "star-rating"), "Zero")

    breadcrumbs = soup.select("ul.breadcrumb li")
    category = breadcrumbs[2].text.strip() if len(breadcrumbs) > 2 else ""

    desc_header = soup.select_one("#product_description")
    description = ""
    if desc_header:
        desc_p = desc_header.find_next_sibling("p")
        if desc_p:
            description = desc_p.text.strip()

    img_tag = soup.select_one("div.item.active img")
    image_url = urljoin(url, img_tag["src"]) if img_tag else ""

    table_data = {}
    for row in soup.select("table.table-striped tr"):
        th = row.select_one("th").text.strip()
        td = row.select_one("td").text.strip()
        table_data[th] = td

    return {
        "url": url,
        "title": title,
        "category": category,
        "rating_raw": rating,
        "description": description,
        "image_url": image_url,
        "upc": table_data.get("UPC", ""),
        "product_type": table_data.get("Product Type", ""),
        "price_excl_tax_raw": table_data.get("Price (excl. tax)", ""),
        "price_incl_tax_raw": table_data.get("Price (incl. tax)", ""),
        "tax_raw": table_data.get("Tax", ""),
        "availability_raw": table_data.get("Availability", ""),
        "num_reviews_raw": table_data.get("Number of reviews", "0")
    }

def extract_all_books(max_workers=20):
    urls = get_all_book_urls()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_books = list(executor.map(fetch_book_detail, urls))
    return raw_books