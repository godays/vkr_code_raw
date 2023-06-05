import os
import json
import pathlib

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from lib.tasks.load_search_results import load_search_results
from lib.tasks.load_page_content import load_page_content
from lib.tasks.url_clustering.url_clustering_main import url_clustering_main


AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '~/airflow')

CFG = {
    "targets_path": "data/{{ ds }}/targets.json",
    "searcher_results_path": "data/{{ ds }}/searcher_results.json",
    "page_content_path": "data/{{ ds }}/page_content.json",
    "failed_to_load_path": "data/{{ ds }}/failed_to_load.json",
    "bert_model_path": "data/{{ ds }}/rubert-tiny",
    "data_output": "data/{{ ds }}/output/",
    "data_output_directory": "data/{{ ds }}/output/clustering",
    "max_page_num": 1,
}

with DAG(
    dag_id="data_loader",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval=None,
) as dag:

    task_targets_file_exists = FileSensor(
        task_id="targets_file_exists",
        filepath=AIRFLOW_HOME + "/" + CFG["targets_path"],
    )

    task_load_search_results = PythonOperator(
        task_id='load_search_results',
        python_callable=load_search_results,
        op_kwargs={
            "targets_path": AIRFLOW_HOME + "/" + CFG["targets_path"],
            "searcher_results_path": AIRFLOW_HOME + "/" + CFG["searcher_results_path"]
        }
    )

    task_load_page_content = PythonOperator(
        task_id='load_page_content',
        python_callable=load_page_content,
        op_kwargs={
            "searcher_results_path": AIRFLOW_HOME + "/" + CFG["searcher_results_path"],
            "page_content_path": AIRFLOW_HOME + "/" + CFG["page_content_path"],
            "failed_to_load_path": AIRFLOW_HOME + "/" + CFG["failed_to_load_path"]
        }
    )

    task_clusterisation = PythonOperator(
        task_id='url_clustering',
        python_callable=url_clustering_main,
        op_kwargs={
            "page_content_path": AIRFLOW_HOME + "/" + CFG["page_content_path"],
            "data_output": AIRFLOW_HOME + "/" + CFG["data_output"],
            "bert_model_path": AIRFLOW_HOME + "/" + "rubert-tiny",
            "data_output_directory": AIRFLOW_HOME + "/" + CFG["data_output_directory"],
            "save_artifacts": True,
        }
    )

    task_targets_file_exists >> task_load_search_results >> task_load_page_content >> task_clusterisation
