#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start point og web scrapper: set up configs and starting loading
"""

import os
import json
import time
import logging
import getpass
from argparse import ArgumentParser
import urllib.parse
from bs4 import BeautifulSoup

from web_scrapper.cfg.web_scrapper_config import WebScrapperConfig
from web_scrapper.data_generator.page_loader import load_page_data
from web_scrapper.data_generator.search_results_loader import search_results_generator
from web_scrapper.data_generator.urls_info_storage import UrlsInfoStorage
from web_scrapper.utils.webdriwer import Webdriwer

logger = logging.getLogger("parser")


def load_phrases():
    logger.info("Loading targets search phrases")
    with (
        open(WebScrapperConfig().raw_targets_path, "r") as raw_targets_file,
        open(WebScrapperConfig().targets_path, "w") as targets_file
    ):
        driver = Webdriwer().driver

        targets = json.load(raw_targets_file)
        url = "https://wordstat.yandex.ru/#!/?words={}".format(urllib.parse.quote(targets[0]))
        driver.get(url)

        login = getpass.getpass("input login for Yandex: ")
        pwd = getpass.getpass("input passeord for Yandex: ")

        inputElement = driver.find_element("css selector", "#b-domik_popup-username")
        inputElement.send_keys(login)

        inputElement = driver.find_element("css selector", "#b-domik_popup-password")
        inputElement.send_keys(pwd)

        driver.find_element("css selector", ".b-domik__button > span").click()

        for target in targets:
            url = "https://wordstat.yandex.ru/#!/?words={}".format(urllib.parse.quote(targets[0]))
            driver.get(url)
            time.sleep(10)

            text = driver.page_source
            soup = BeautifulSoup(text, "html.parser")

            phrases = [phrase.getText().lower() for phrase in soup.findAll(".b-word-statistics__td-phrase > span > a")]
            targets_file.write(
                json.dumps(
                    {
                        "target": target, 
                        "phrases": [phrase for phrase in phrases if phrase]
                    }
                )
            )

def load_search_reults():
    logger.info("Loading search results")
    with (
        open(WebScrapperConfig().targets_path, "r") as targets_file, 
        open(WebScrapperConfig().searcher_results_path, "w") as searcher_results,
        UrlsInfoStorage()
    ):
        for target_line in targets_file.readlines():
            target_cfg = json.loads(target_line)
            target = target_cfg["target"]
            for target_phrase in target_cfg["phrases"]:
                for record in search_results_generator(target_phrase):
                    record["target"] = target
                    searcher_results.write(json.dumps(record) + '\n')
        
def load_pages():
    logger.info("Loading page content")
    with (
        open(WebScrapperConfig().searcher_results_path, "r") as searcher_results,
        open(WebScrapperConfig().pages_path, "a") as pages,
        UrlsInfoStorage()
    ):
        for line in searcher_results.readlines():
            record = json.loads(line)
            page_data = load_page_data(record)
            if page_data is not None:
                if not page_data["url"] in UrlsInfoStorage().already_loaded:
                    pages.write(json.dumps(page_data) + '\n')
                    UrlsInfoStorage().already_loaded.add(page_data["url"])

def load(config_path: str = None) -> None:
    """
    :param config_path: str - scrapper cfg path
    Main behaviour implementation
    """
    WebScrapperConfig().setup(config_path)
    #if not os.path.exists(WebScrapperConfig().targets_path):
    #   load_phrases()

    if not os.path.exists(WebScrapperConfig().searcher_results_path):
        load_search_reults()
    
    if not os.path.exists(WebScrapperConfig().pages_path):
        load_pages()


def setup_argparser() -> ArgumentParser:
    """
    :return: ArgumentParser - configured ArgumentParser
    Setup ArgumentParser
    """
    parser = ArgumentParser(
        prog="web scrapper",
        description="Create datasets by request"
    )

    parser.add_argument("-c", "--config_path", dest="config_path", type=str)

    return parser


if __name__ == "__main__":
    argparser = setup_argparser()
    args = argparser.parse_args()

    load(args.config_path)
