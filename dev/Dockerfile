# Dev/labelling image: mirrors the environment the organisers labelled in.
# Not the submission image. Build: docker build -f dev/Dockerfile -t br-dev .
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
RUN git init -q /opt/sqlglot && cd /opt/sqlglot \
 && git remote add origin https://github.com/tobymao/sqlglot.git \
 && git fetch -q --depth 1 origin eb11aec4be8cb2122afe713a4816662c3f0fb2fa \
 && git checkout -q FETCH_HEAD && rm -rf .git
ENV PYTHONPATH=/opt/sqlglot SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/sqlglot
