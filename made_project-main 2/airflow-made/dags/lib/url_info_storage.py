#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wrapper over file that contain urls that can be loaded only with selenium
"""

import json
import os
from web_scrapper.cfg.web_scrapper_config import WebScrapperConfig
from web_scrapper.utils.singleton import singleton


@singleton
class UrlsInfoStorage:
    def __init__(self):
        self.selenium_urls = None
        self.failing_urls = None
        self.already_loaded = set()

        self.pages_path = pages_path
        self.selenium_urls_path
        self.failing_urls_path

        if os.path.exists(WebScrapperConfig().pages_path):
            with open(WebScrapperConfig().pages_path, "r") as pages_file:
                for line in pages_file.readlines():
                    try:
                        url = json.loads(line)["url"]
                    except:
                        pass
                    self.already_loaded.add(url)

    def __enter__(self):
        self.selenium_urls = set()
        if os.path.exists(WebScrapperConfig().selenium_urls_path):
            with open(WebScrapperConfig().selenium_urls_path, "r+") as urls_file:
                try:
                    self.selenium_urls = set(json.load(urls_file))
                except:
                    pass


        self.failing_urls = set()
        if os.path.exists(WebScrapperConfig().failing_urls_path):
            with open(WebScrapperConfig().failing_urls_path, "r+") as urls_file:
                try:
                    self.failing_urls = set(json.load(urls_file))
                except:
                    pass

    def __exit__(self, *args, **kwargs):
        with open(WebScrapperConfig().selenium_urls_path, "w+") as urls_file:
            json.dump(list(self.selenium_urls), urls_file)
            self.selenium_urls = None
        
        with open(WebScrapperConfig().failing_urls_path, "w+") as urls_file:
            json.dump(list(self.failing_urls), urls_file)
            self.failing_urls = None
