#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate search requests for target phrase
"""
import os
import logging
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from web_scrapper.cfg.web_scrapper_config import WebScrapperConfig
from web_scrapper.utils.singleton import singleton


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger("parser")

@singleton
class Webdriwer:

    def __init__(self, headless="phantomjs", browser="chrome", webdriver_path=None, base_path="./web_scrapper"):
        """
        Args:
            headless (str): Headless tool name.
            browser (str): Browser to run selenium.
            webdriver_path (str): Specific web driver path other than default.
        """

        self.platform = "linux"
        if sys.platform == "win32":
            self.platform = "windows"
        elif sys.platform == "darwin":
            self.platform = "mac"

        if browser:
            self.webdriver_path = os.path.join(
                base_path, "webdrivers", self.platform, WebScrapperConfig().webdriver_settings[browser][self.platform]
            )
        elif webdriver_path:
            self.webdriver_path = webdriver_path
        else:
            self.webdriver_path = os.path.join(
                BASE_DIR, "webdrivers", self.platform, WebScrapperConfig().webdriver_settings[headless][self.platform]
            )

        logger.info("Current webdriver_path: {}".format(self.webdriver_path))
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--remote-debugging-port=9222")

        if browser == "chrome":
            self.driver = webdriver.Chrome(self.webdriver_path, chrome_options=chrome_options)
        elif browser == "firefox":
            self.driver = webdriver.Firefox(self.webdriver_path)
        else:
            # ToDo???
            #self.driver = webdriver.PhantomJS(self.webdriver_path)
            self.driver = webdriver.Chrome(self.webdriver_path)
        self.driver.implicitly_wait(30)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.driver.quit()

    def get_page(self, url):
        self.driver.get(url)
        return self.driver.page_source