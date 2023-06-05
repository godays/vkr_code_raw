FROM python:3.9
COPY setup.py /home/
COPY app /home/

WORKDIR /home
RUN pip3 install -e .
COPY requirements.txt /home/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r /home/requirements.txt
run pip install -U tokenizers

RUN pip install --upgrade pip
RUN pip install transformers
RUN pip install tensorflow==2.7.0

RUN pip install -U sentencepiece==0.1.95
RUN pip install -U pandas==1.1.5
RUN pip install -U numpy==1.21.6
RUN pip install -U scikit-learn==1.0.2
RUN pip install -U tokenizers>0.11

ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0
