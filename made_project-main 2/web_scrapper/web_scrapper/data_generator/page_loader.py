#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start point og web scrapper: set up configs and starting loading
"""

import re
from typing import List, Dict
import requests
from urllib.parse import urlparse
import logging
from bs4 import BeautifulSoup
from web_scrapper.data_generator.urls_info_storage import UrlsInfoStorage
from web_scrapper.utils.webdriwer import Webdriwer


logger = logging.getLogger("parser")

def prepare_text(phrase: str) -> List[str]:
    return re.sub(r"[^\w]+", " ", phrase.lower())

def load_page(url: str, descr: List[str]) -> BeautifulSoup:
    """
    :param url: str - url of loading page
    :param descr: arr[str] - 
    """

    # requests load
    base_path = urlparse(url).netloc
    if base_path in UrlsInfoStorage().failing_urls:
        return None
    if not base_path in UrlsInfoStorage().selenium_urls:
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
        text = Webdriwer().get_page(url)
        soup = BeautifulSoup(text, "html.parser")
        text = soup.getText().strip()
        cleaned_text = prepare_text(text)

        for phrase in descr:
            if prepare_text(phrase) in cleaned_text:
                if not base_path in UrlsInfoStorage().selenium_urls:
                    UrlsInfoStorage().selenium_urls.add(base_path)
                return soup

    except Exception as err:
        logger.error("Failed to get url {} with err: {}".format(url, err))
        UrlsInfoStorage().failing_urls.add(url)
    return None

def load_page_data(url_data: Dict[str, str]):
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
    soup = load_page(url_data["url"], url_data["url_descr"])
    if soup is None:
        logger.info("Failed to load {}".format(url_data["url"]))
        return None
    logger.info("Loaded {}".format(url_data["url"]))

    #url_data["title"] = ???
    #url_data["descr"] = ???
    #url_data["links"] = ???
    url_data["html"] = soup.text
    url_data["text"] = soup.getText()
    url_data["unique_words"] = list(set(prepare_text(soup.getText()).split(" ")))
    url_data["h1_heading"] = [tag.getText() for tag in soup.find_all("h1")]
    url_data["h2_heading"] = [tag.getText() for tag in soup.find_all("h2")]
    url_data["h3_heading"] = [tag.getText() for tag in soup.find_all("h3")]
    url_data["bold_text"] = [tag.getText() for tag in soup.find_all("b")]

    return url_data
