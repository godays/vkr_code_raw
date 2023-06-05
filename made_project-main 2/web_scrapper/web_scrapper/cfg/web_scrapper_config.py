#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup logger configuration from cfg_path arg (default is in BASE_CFG_PATH)
"""

import logging
import logging.config
import yaml

from web_scrapper.utils.singleton import singleton

BASE_CFG_PATH = "./web_scrapper/cfg/cfg.yml"


@singleton
class WebScrapperConfig:
    """
    Singleton web scrapper config source
    """

    target_path = None
    parsers = {}

    def setup(self, cfg_path: str):
        """
        :param cfg_path: str - path to yaml cfg file
        - setup logging conf
        - getting cfg fields from cfg_path
        """
        if cfg_path is None:
            cfg_path = BASE_CFG_PATH

        # load cfg
        with open(cfg_path) as config_fin:
            cfg = yaml.safe_load(config_fin)

        if cfg is None:
            raise Exception("Failed to load parser Cfg")
        # logger setuping
        self.setup_logging(cfg["cfg"].get("logger_cfg_path"))

        # main cfgs:
        self.logger_cfg_path = cfg["cfg"]["logger_cfg_path"]
        self.raw_targets_path = cfg["cfg"]["raw_targets_path"]
        self.targets_path = cfg["cfg"]["targets_path"]
        self.failing_urls_path = cfg["cfg"]["failing_urls_path"]
        self.searcher_results_path = cfg["cfg"]["searcher_results_path"]
        self.selenium_urls_path = cfg["cfg"]["selenium_urls_path"]
        self.pages_path = cfg["cfg"]["pages_path"]
        self.max_page_num = cfg["cfg"]["max_page_num"]

        self.webdriver_settings = cfg["web_driver_cfg"]

    def setup_logging(self,logging_yaml_config_fpath) -> None:
        """
        :param cfg_path: str - path to logger cfg file
        setup logging via YAML if it is provided
        """
        if logging_yaml_config_fpath:
            with open(logging_yaml_config_fpath) as config_fin:
                logging.config.dictConfig(yaml.safe_load(config_fin))
