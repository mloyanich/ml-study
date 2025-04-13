import argparse
import csv
import json
import os
import random
import time
from selenium import webdriver
from bs4 import BeautifulSoup


def scrape_carvana_page(page_num, csv_filename="carvana_cars.csv"):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    options.add_argument("--headless")  # Run in headless mode (no UI)
    options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    options.add_argument(
        "--no-sandbox"
    )  # Needed for headless mode in some environments

    driver = webdriver.Chrome(options=options)
    url = f"https://www.carvana.com/cars?page={page_num}"
    print(f"Scraping page {page_num}...")

    driver.get(url)
    time.sleep(random.randint(8, 12))  # Give Cloudflare a chance to finish

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Extract the JSON-LD data from the script tags
    scripts = soup.find_all(
        "script", {"data-qa": "vehicle-ld", "type": "application/ld+json"}
    )
    cars = []

    for script in scripts:
        try:
            data = json.loads(script.string)
            car = {
                "name": data.get("name"),
                "model": data.get("model"),
                "brand": data.get("brand"),
                "year": data.get("name").split(" ")[0],
                "price": data.get("offers", {}).get("price"),
                "mileage": data.get("mileageFromOdometer"),
                "url": data.get("offers", {}).get("url"),
            }
            cars.append(car)
        except Exception as e:
            print("Parse error:", e)
    if not cars:
        print(
            "❌ No car data found. Cloudflare might’ve blocked the page or the page is empty."
        )
        return

    # Load existing CSV to check for duplicates
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_urls = {row["url"] for row in reader}
    else:
        existing_urls = set()

    # Append new rows to CSV, skipping duplicates
    preferred_order = ["brand", "model", "year", "mileage", "price", "name", "url"]
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred_order)
        if f.tell() == 0:
            writer.writeheader()  # Write header only if the file is empty
        new_cars_count = 0
        for car in cars:
            if car["url"] not in existing_urls:  # Skip duplicates based on URL
                writer.writerow(car)
                existing_urls.add(car["url"])
                new_cars_count += 1
    print(f"✅ Added {new_cars_count} new cars to {csv_filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Carvana cars for a specific page."
    )
    parser.add_argument(
        "--page",
        type=int,
        help="Page number to scrape (optional). If not provided, picks one randomly.",
    )
    args = parser.parse_args()

    page_to_scrape = args.page if args.page else random.randint(1, 130)
    scrape_carvana_page(page_to_scrape)
