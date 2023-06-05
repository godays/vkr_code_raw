#!/usr/bin/env bash

airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Anonymous \
    --lastname Admin \
    --role Admin \
    --email admin@example.org
airflow connections add 'fs_default' --conn-type 'file' --conn_extra '{\"autoconfig\": true}'
