#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start point of web scrapper: set up configs and starting loading
"""

import logging
import json 
from lib.parsers.google_parser import GoogleParser

logger = logging.getLogger("parser")


def phrase_generator(target_phrase):
    yield target_phrase

def search_results_generator(target_phrase, max_page_num):
    for search_text in phrase_generator(target_phrase):
        for searcher_parser in [GoogleParser()]:
            for search_result in searcher_parser.generate_search_results(search_text, max_page_num):
                search_result["target"] = target_phrase
                logger.info(search_result)

                yield search_result

def load_search_results(
    targets_path, 
    searcher_results_path,
    max_page_num=1
):
    logger.info("Loading search results")
    writed_urls = set()
    
    import os
    airflow_home= os.environ.get('AIRFLOW_HOME', '~/airflow')
    logger.info("fs: {}".format(os.listdir(airflow_home)))

    i = 0

    with open(targets_path, "r") as targets_file, open(searcher_results_path, "w") as searcher_results:
        for target_line in targets_file.readlines():
            target_cfg = json.loads(target_line)
            target = target_cfg["target"]
            logger.info("Processing target '{}'".format(target))
            for target_phrase in target_cfg["phrases"]:
                for record in search_results_generator(target_phrase, max_page_num):
                    record["target"] = target
                    if record["url"] in writed_urls:
                        continue
                    writed_urls.add(record["url"])
                    searcher_results.write(json.dumps(record) + '\n')
                    i += 1
                    # DEBUG!
                    if i > 20:
                        return
