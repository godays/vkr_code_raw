#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class for parse google results
"""

import urllib
from urllib.parse import unquote
from web_scrapper.parsers.base_parser import BaseParcer, parcer
from bs4 import BeautifulSoup

GOOGLE_URL_REQUEST_TEMPLATE = "https://www.google.com/search?q={request_text}&aqs=chrome..69i57j0i22i30l9.9960j0j7&client=ubuntu&sourceid=chrome&ie=UTF-8&start={start_idx}"


@parcer("Google")
class GoogleParser(BaseParcer): 

    def build_url_for_search_text(self, search_text: str, page_idx: int = 0) -> str:
        """
        Return request url for searcing text
        """
        return GOOGLE_URL_REQUEST_TEMPLATE.format(
            request_text=urllib.parse.quote(search_text),
            start_idx=page_idx * 10
        )
    
    def extract_title_and_content(self, block):
        block_children = list(block.children)
        if len(block_children) != 2 or block_children[0].name != "div" or block_children[1].name != "div":
            raise Exception()
        
        return block_children[0], block_children[1]

    def parse_content_block(self, content_block):
        return content_block.getText()

    def parse_title_block(self, title_block):
        link_block = title_block.findChild("a")
        
        url_href = link_block.attrs["href"].lstrip("/url=?q=")
        url_parts = url_href.split("&") # google adds its own arguments
        url = url_parts[0]
        for part in url_parts[1:]:
            if (
                part.startswith("sa") or
                part.startswith("usg") or
                part.startswith("ved")
            ):
                break

            url += "&" + part
        url = unquote(url)

        url_text = link_block.findChild("h3").getText()
        site_path = None
        try:
            site_path = link_block.select("div > div > div").getText()
        except:
            pass
        
        return url, url_text, site_path

    def generate_search_results_from_page(self, soup):
        main_block = soup.find("div", id="main")
        for block in main_block.children:
            try:
                if block.name != "div":
                    continue

                block_children = list(block.children)
                if len(block_children) != 1 or block_children[0].name != "div":
                    continue

                title_block, content_block = self.extract_title_and_content(block_children[0])

                url, url_text, site_path = self.parse_title_block(title_block)
                url_descr = self.parse_content_block(content_block)

                yield{
                    "url": url,
                    "url_text": url_text,
                    "site_path": site_path,
                    "url_descr": url_descr
                }
            except:
                continue

    def parse_has_next(self, soup):
        try:
            return soup.find("footer").find("a").getText().lower().startswith("след")
        except:
            pass

        return False
