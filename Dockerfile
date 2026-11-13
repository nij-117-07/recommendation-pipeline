ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.12
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.12
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir \
    -r /tmp/requirements-docker.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
