#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start point og web scrapper: set up configs and starting loading
"""

import logging
import requests
from typing import Dict
from web_scrapper.cfg.web_scrapper_config import WebScrapperConfig
from web_scrapper.parsers import *

logger = logging.getLogger("parser")


def phrase_generator(target_phrase):
    yield target_phrase

def search_results_generator(target_phrase):
    for search_text in phrase_generator(target_phrase):
        for searcher_parser in WebScrapperConfig().parsers.values():
            for search_result in searcher_parser.generate_search_results(search_text, WebScrapperConfig().max_page_num):
                search_result["target"] = target_phrase
                logger.info(search_result)

                yield search_result
