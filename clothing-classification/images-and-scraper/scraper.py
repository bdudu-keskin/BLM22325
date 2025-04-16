from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
import requests
import time
import concurrent.futures
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class ClothingScraper:
    def __init__(self, base_url):
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-images')
        self.base_url = base_url
        self.driver = webdriver.Chrome(options=options)
        self.categories = {
            'üst_giyim': ['tişört', 'gömlek', 'kazak'],
            'alt_giyim': ['pantolon', 'etek', 'şort'],
            'diğer': ['çanta', 'ayakkabı', 'şapka']
        }

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.download_count = 0
        self.download_lock = threading.Lock()

    def try_click_next_page(self):
        next_page_selectors = [
            "a.next",
            "a.pagination-next",
            "a[rel='next']",
            "li.next a",
            "a[aria-label='Next']",
            "button.next-page",
            "//a[contains(text(), 'Next')]",
            "//a[contains(text(), 'Sonraki')]",
            "//button[contains(text(), 'Next')]",
            "//button[contains(text(), 'Sonraki')]"
        ]

        for selector in next_page_selectors:
            try:
                if selector.startswith("//"):
                    next_button = self.driver.find_element(By.XPATH, selector)
                else:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)

                if next_button.is_displayed() and next_button.is_enabled():
                    next_button.click()
                    time.sleep(2)
                    return True
            except NoSuchElementException:
                continue
            except Exception as e:
                print(f"Error clicking next page: {e}")
                continue

        return False

    def try_url_pagination(self, current_url, page):
        url_patterns = [
            f"{current_url}&page={page}",
            f"{current_url}?page={page}",
            f"{current_url}/page/{page}",
            current_url.replace(f"page={page - 1}", f"page={page}")
        ]

        original_url = self.driver.current_url

        for url in url_patterns:
            try:
                self.driver.get(url)
                time.sleep(2)

                if self.driver.current_url != original_url:
                    return True
            except:
                continue

        return False

    def scroll_to_bottom(self):
        SCROLL_PAUSE_TIME = 0.5
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def create_directories(self):
        if not os.path.exists('clothing_images'):
            os.makedirs('clothing_images')

        for category, items in self.categories.items():
            category_path = f'clothing_images/{category}'
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            for item in items:
                item_path = f'{category_path}/{item}'
                if not os.path.exists(item_path):
                    os.makedirs(item_path)

    def download_image(self, url, category, item_type, index):
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                file_path = f'clothing_images/{category}/{item_type}/image_{index}.jpg'
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                with self.download_lock:
                    self.download_count += 1
                    if self.download_count % 10 == 0:
                        print(f"Downloaded {self.download_count} images...")
                return True
        except Exception as e:
            print(f"Error downloading image: {e}")
        return False

    def download_images_parallel(self, image_data):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for url, category, item_type, index in image_data:
                future = executor.submit(self.download_image, url, category, item_type, index)
                futures.append(future)
            concurrent.futures.wait(futures)

    def scrape_page(self, category, item_type, base_index=0):
        try:
            print(f"Scraping page...")
            time.sleep(2)

            self.scroll_to_bottom()

            selectors = [
                "img.product-image",
                "img.lazy",
                "img[data-src]",
                "img.product-img",
                "img.product",
                "div.product img",
                "img[src*='product']",
                "img"
            ]

            images = []
            for selector in selectors:
                try:
                    images = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if images:
                        print(f"Found {len(images)} images with selector: {selector}")
                        break
                except:
                    continue

            if not images:
                print(f"No images found on this page")
                return 0

            image_data = []
            for index, img in enumerate(images, start=base_index):
                try:
                    image_url = img.get_attribute('src') or img.get_attribute('data-src') or img.get_attribute(
                        'data-lazy-src')

                    if image_url:
                        if image_url.startswith('//'):
                            image_url = 'https:' + image_url
                        elif image_url.startswith('/'):
                            image_url = self.base_url + image_url

                        image_data.append((image_url, category, item_type, index))
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue

            self.download_images_parallel(image_data)
            return len(images)

        except Exception as e:
            print(f"Error scraping page: {e}")
            return 0

    def scrape_category(self, category, item_type):
        search_url = f"{self.base_url}/search?q={item_type}"
        self.driver.get(search_url)

        page = 21
        total_images = 0
        max_pages = 40  # Safety limit

        while page <= max_pages:
            print(f"\nScraping page {page} for {item_type}...")
            images_found = self.scrape_page(category, item_type, total_images)

            if images_found == 0:
                print(f"No images found on page {page}, stopping pagination")
                break

            total_images += images_found

            # Try next page button first, then URL pagination
            if not self.try_click_next_page():
                if not self.try_url_pagination(search_url, page + 1):
                    print(f"No more pages found for {item_type}")
                    break

            page += 1
            time.sleep(2)

    def scrape_all(self):
        self.create_directories()

        for category, items in self.categories.items():
            for item in items:
                print(f"\nScraping {item} in category {category}")
                self.scrape_category(category, item)

    def cleanup(self):
        self.driver.quit()


def main():
    base_url = "https://boyner.com.tr"  # Replace with actual website URL

    scraper = ClothingScraper(base_url)
    try:
        scraper.scrape_all()
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()