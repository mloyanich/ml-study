import argparse
import csv
import json
import requests
import os
import random
import time
from selenium import webdriver
from bs4 import BeautifulSoup

example = {
    "@context": "https://schema.org",
    "@type": "Vehicle",
    "itemCondition": "Used",
    "name": "2020 Chevrolet Spark",
    "modelDate": 2020,
    "manufacturer": "Chevrolet",
    "model": "Spark",
    "color": "Other",
    "image": "https://cdnblob.fastly.carvana.io/2003435545/post-large/normalized/zoomcrop/2003435545-edc-02.jpg?v=2025.4.13_1.13.48",
    "brand": "Chevrolet",
    "description": "Used 2020 Chevrolet Spark LS with 12361 miles - $14,990",
    "mileageFromOdometer": 12361,
    "sku": 2003435545,
    "vehicleIdentificationNumber": "KL8CB6SA2LC430091",
    "offers": {
        "@type": "Offer",
        "price": 14990,
        "priceCurrency": "USD",
        "availability": "http://schema.org/InStock",
        "priceValidUntil": "January 1, 2030",
        "url": "https://www.carvana.com/vehicle/3457647",
    },
}


def get_vin_data(vin):
    url = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
    post_fields = {"format": "json", "data": vin}
    response = requests.post(url, data=post_fields, timeout=10)
    result = json.loads(response.text)
    return result["Results"][0]


def scrape_carvana_page(page_num, csv_filename="carvana_cars.csv"):
    print(f"Scraping page {page_num}...")
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

    driver.get(url)
    time.sleep(random.randint(8, 12))  # Give Cloudflare a chance to finish

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Extract the JSON-LD data from the script tags
    scripts = soup.find_all(
        "script", {"data-qa": "vehicle-ld", "type": "application/ld+json"}
    )
    cars = []
    today = time.strftime("%Y-%m-%d")
    for script in scripts:
        try:
            data = json.loads(script.string)
            car = {
                "name": data.get("name"),
                "model": data.get("model"),
                "brand": data.get("brand"),
                "year": data.get("modelDate"),
                "price": data.get("offers", {}).get("price"),
                "mileage": data.get("mileageFromOdometer"),
                "url": data.get("offers", {}).get("url"),
                "image": data.get("image"),
                "color": data.get("color"),
                "description": data.get("description"),
                "sku": data.get("sku"),
                "vin": data.get("vehicleIdentificationNumber"),
                "condition": data.get("itemCondition"),
                "scraped_at": today,
            }
            print(car)
            cars.append(car)
        except Exception as e:
            print("Parse error:", e)
    if not cars:
        print(
            "❌ No car data found. Cloudflare might’ve blocked the page or the page is empty."
        )
        return
    print(f"found {len(cars)} cars")
    # Load existing CSV to check for duplicates
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_urls = {row["url"] for row in reader}
    else:
        print("no csv file found, creating new one")
        existing_urls = set()
    full_car_keys = {**cars[0], **get_vin_data(cars[0]["vin"])}.keys()

    # Append new rows to CSV, skipping duplicates
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=full_car_keys)
        if f.tell() == 0:
            writer.writeheader()  # Write header only if the file is empty
        new_cars_count = 0
        for car in cars:
            if car["url"] not in existing_urls:  # Skip duplicates based on URL
                full_car = {**car, **get_vin_data(car["vin"])}
                writer.writerow(full_car)
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

    page_to_scrape = args.page if args.page else random.randint(1, 100)
    last_page = page_to_scrape + 3
    try:
        while page_to_scrape <= last_page:
            scrape_carvana_page(page_to_scrape)
            page_to_scrape += 1
            print(f"Page {page_to_scrape} scraped. Sleeping for a bit...")
            time.sleep(random.randint(5, 10))
    except Exception as e:
        print(f"Scraping interrupted. {e}")
