import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://www.shl.com/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_assessment_links():
    all_links = []
    
    # Types 1 (Knowledge & Skills) and 2 (Personality & Behavior)
    # Based on initial exploration:
    # Type 1 has up to start=372 (32 pages)
    # Type 2 has up to start=132 (12 pages)
    
    for t in [1, 2]:
        start = 0
        while True:
            url = f"{BASE_URL}?start={start}&type={t}"
            logging.info(f"Fetching list page: {url}")
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                if response.status_code != 200:
                    logging.error(f"Failed to fetch {url}: Status {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all links that look like assessment views
                links = soup.select('a[href*="/products/product-catalog/view/"]')
                if not links:
                    logging.info(f"No more links found for type {t} at start {start}")
                    break
                
                for link in links:
                    href = link['href']
                    if not href.startswith('http'):
                        href = "https://www.shl.com" + href
                    
                    if href not in [l['url'] for l in all_links]:
                        all_links.append({
                            'name': link.get_text(strip=True),
                            'url': href,
                            'test_type': 'K' if t == 1 else 'P'
                        })
                
                # Check for "Next" button or just increment start by 12
                # From exploration, items per page is 12
                start += 12
                
                # Safety break if we get too many (unlikely but good practice)
                if start > 500:
                    break
                    
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                logging.error(f"Error fetching {url}: {e}")
                break
                
    logging.info(f"Found {len(all_links)} total assessment links.")
    return all_links

def get_assessment_details(links):
    data = []
    for i, item in enumerate(links):
        url = item['url']
        logging.info(f"[{i+1}/{len(links)}] Fetching details for: {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                logging.error(f"Failed to fetch {url}: Status {response.status_code}")
                data.append({
                    'assessment_name': item['name'],
                    'assessment_url': url,
                    'description': '',
                    'test_type': item['test_type']
                })
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This part depends on the exact HTML structure
            # Based on standard SHL catalog pages:
            # Name: h1
            # Description: div with class "description" or "rich-text"
            
            name = soup.find('h1').get_text(strip=True) if soup.find('h1') else item['name']
            
            # Often the description is in a specific div
            desc_div = soup.find('div', class_='rich-text') or soup.find('div', class_='description')
            if not desc_div:
                # Fallback: look for the first few paragraphs
                paragraphs = soup.find_all('p')
                description = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])
            else:
                description = desc_div.get_text(strip=True)
            
            data.append({
                'assessment_name': name,
                'assessment_url': url,
                'description': description,
                'test_type': item['test_type']
            })
            
            # Save every 10 items just in case
            if (i + 1) % 10 == 0:
                pd.DataFrame(data).to_csv("shl_catalogue_partial.csv", index=False)
                
            time.sleep(random.uniform(0.2, 0.5))
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            data.append({
                'assessment_name': item['name'],
                'assessment_url': url,
                'description': '',
                'test_type': item['test_type']
            })
            
    return data

if __name__ == "__main__":
    logging.info("Starting SHL Catalog Crawler...")
    links = get_assessment_links()
    if links:
        details = get_assessment_details(links)
        df = pd.DataFrame(details)
        df.to_csv("shl_catalogue.csv", index=False)
        logging.info(f"Crawler finished. Saved {len(df)} records to shl_catalogue.csv")
    else:
        logging.error("No assessment links found.")
