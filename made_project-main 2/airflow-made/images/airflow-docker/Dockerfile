ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==2.6.0

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages
ENV PYTHONUNBUFFERED 1

# init airflow database
COPY entrypoint.sh /entrypoint.sh
RUN chmod u+x /entrypoint.sh

# install google chrome
RUN apt-get update \
    && apt-get -y install curl unzip
RUN curl https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/
ENV CHROMEDRIVER /usr/local/bin/chromedriver

# post install selenium
USER airflow
RUN pip install --user --no-cache-dir selenium

COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
