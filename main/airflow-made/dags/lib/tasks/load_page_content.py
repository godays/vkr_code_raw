#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loading page content via requests or selenium
"""

import os
import logging
import json 
import re
import requests
from typing import List, Dict
from urllib.parse import urlparse
import logging
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver


logger = logging.getLogger("parser")

class Webdriwer:
    def __enter__(self):
        self.webdriver_path = os.environ.get('CHROMEDRIVER_PATH', '/usr/local/bin/chromedriver')

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--remote-debugging-port=9222")

        self.driver = webdriver.Chrome(self.webdriver_path, chrome_options=chrome_options)
        self.driver.implicitly_wait(30)

        return self

    def __exit__(self, type, value, tb):
        self.driver.quit()

    def get_page(self, url):
        self.driver.get(url)
        return self.driver.page_source

def prepare_text(phrase: str) -> List[str]:
    return re.sub(r"[^\w]+", " ", phrase.lower())

def load_page(url: str, descr: List[str], failing_urls, selenium_urls) -> BeautifulSoup:
    """
    :param url: str - url of loading page
    :param descr: arr[str] - 
    """

    # requests load
    base_path = urlparse(url).netloc
    if base_path in failing_urls:
        return None
    if not base_path in selenium_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.getText().strip()

            cleaned_text = prepare_text(text)
            for phrase in descr:
                if prepare_text(phrase) in cleaned_text:
                    return soup

        except Exception as err:
            logger.error("Failed to get url {} with err: {}".format(url, err))

    # selenium load
    try:
        logger.error("Geting page content {} by selenium".format(url, err))
        text = Webdriwer().get_page(url)
        soup = BeautifulSoup(text, "html.parser")
        text = soup.getText().strip()
        cleaned_text = prepare_text(text)

        for phrase in descr:
            if prepare_text(phrase) in cleaned_text:
                if not base_path in selenium_urls:
                    selenium_urls.add(base_path)
                return soup

    except Exception as err:
        logger.error("Failed to get url {} by selenium with err: {}".format(url, err))
        failing_urls.add(url)
    return None

def load_page_data(url_data: Dict[str, str], failing_urls, selenium_urls):
    """
    :param: Dict[str, str] - search data {
        "url": url - by this url we are getting info
        "url_text": str - this text search in page
        "site_path": str, - 
        "url_descr":  str
    }
    :return: Dict[str, str] {
        "url": url
        "url_text": str,
        "site_path": str,
        "url_descr":  str
        URL
        TITLE
        DESCRIPTION
        ALL WORDS
        UNIQUE KEYWORDS
        H1 HEADINGS
        H2 HEADINGS
        H3 HEADINGS
        BOLD TEXT
        ON PAGE LINKS
        PAGE TEXT
    }
    """

    logger.info("Loading {} ...".format(url_data["url"]))
    soup = load_page(url_data["url"], url_data["url_descr"], failing_urls, selenium_urls)
    if soup is None:
        logger.info("Failed to load {}".format(url_data["url"]))
        return None
    logger.info("Loaded {}".format(url_data["url"]))

    url_data["html"] = soup.text
    url_data["text"] = soup.getText()
    url_data["unique_words"] = list(set(prepare_text(soup.getText()).split(" ")))
    url_data["h1_heading"] = [tag.getText() for tag in soup.find_all("h1")]
    url_data["h2_heading"] = [tag.getText() for tag in soup.find_all("h2")]
    url_data["h3_heading"] = [tag.getText() for tag in soup.find_all("h3")]
    url_data["bold_text"] = [tag.getText() for tag in soup.find_all("b")]

    return url_data

def load_page_content(
    searcher_results_path,
    page_content_path,
    failed_to_load_path
):

    failing_urls = set()
    selenium_urls = set()

    last_url = None
    if os.path.exists(page_content_path):
        with open(page_content_path, "r") as f:
            for line in f:
                last_url = line
        if last_url is not None:
            last_url = json.loads(last_url)["url"]

    with open(searcher_results_path, "r") as searcher_results, \
        open(page_content_path, "a") as pages, \
        open(failed_to_load_path, "a") as failed_to_load, \
        Webdriwer():

        if last_url is not None:
            for line in searcher_results.readlines():
                record = json.loads(line)
                if record["url"] == last_url:
                    break

        for line in searcher_results.readlines():
            record = json.loads(line)

            logger.info("Loading page content for {}...".format(record["url"]))
            page_data = load_page_data(record, failing_urls, selenium_urls)
            if page_data is not None:
                logger.info("Successful loaded page content for {}".format(record["url"]))
                pages.write(json.dumps(page_data) + '\n')
            else:
                logger.info("Faailed to load page content for {}".format(record["url"]))
                failed_to_load.write(record["url"])


