#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base class for searcher parsers
"""

import requests
import logging
from typing import Dict, Generator
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup

logger = logging.getLogger("parser")


def parcer(parser_name):
    def wrapper(parser_class):
        parser_class.parser_type = parser_name
        return 
    return wrapper

class BaseParcer:
    """
    Implement build_url_for_search_text
    """

    parser_class = None

    def build_url_for_search_text(self, search_text: str, page_idx: int = 0) -> str:
        """
        :param search_text: str - text for searching withoout encoding
        :param page_idx: int - index of page
        :return: str - searcher url
        Return request url for searcing text
        """
        raise NotImplementedError()

    def generate_search_results_from_page(self, soup):
        """
        :param soup: BeautifulSoup - bs wrapper over page content
        :return: Generator[Dict[str, str]] - Generator of search results with structure:
            {
                "url": urlc,
                "url_text": url_text: str,
                "site_path": site_path: str,
                "url_descr": url_descr: str
            }
        """
        raise NotImplementedError()
    
    def parse_has_next(self, soup) -> bool:
        """
        :return: bool - has next page
        """
        raise NotImplementedError()

    def get_response(self, search_text, page_idx):
        """
        Load page
        """
        try:
            url = self.build_url_for_search_text(search_text, page_idx)
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            logger.error("Failed to load url {}, with err {}".format(url, err))

        return response

    def generate_search_results(self, search_text, page_limit=10):
        """
        :return: Generator[Dict(str, str)] - Generator of search results with structure:
            {
                "url": urlc,
                "url_text": url_text: str,
                "site_path": site_path: str,
                "url_descr": url_descr: str
            }
        """
        for page_idx in range(page_limit):

            response = self.get_response(search_text, page_idx)
            if response is None:
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            for record in self.generate_search_results_from_page(soup):
                record["searcher"] = self.parser_type
                record["search_phrase"] = search_text
                yield record
            
            has_next = self.parse_has_next(soup)
            if not has_next:
                break
