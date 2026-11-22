# Blast Radius submission: static test selector + in-container verification (phase 4).
# The digest is the image the organisers labelled in (keep it: its Python decides test behaviour).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The labelling environment: the exact lock file from the starter kit.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

# sqlglot at the pinned commit, from sqlglot's own repository (the slim image has no git).
# Precompiled in timestamp mode: the verifier's overlay reuses these .pyc for unchanged
# modules, and Python recompiles any patched file because its mtime differs.
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN python -c "import io, tarfile, urllib.request; \
url = 'https://github.com/tobymao/sqlglot/archive/${SQLGLOT_COMMIT}.tar.gz'; \
data = urllib.request.urlopen(url, timeout=120).read(); \
tarfile.open(fileobj=io.BytesIO(data)).extractall('/opt', filter='data')"
RUN mv /opt/sqlglot-${SQLGLOT_COMMIT} /opt/sqlglot \
 && python -m compileall -q -j 4 /opt/sqlglot/sqlglot /opt/sqlglot/tests \
 && chmod -R a+rX /opt/sqlglot

# app/  the selector and verifier (stdlib only in the selector)
# maps/ built offline from sqlglot @ eb11aec by tools/make_maps.sh (per-test coverage,
#       import contexts, class usage, data files, collection-time imports). See CHANGELOG.md.
COPY app/ /app/
COPY maps/ /app/maps/
# Executor-test caches (app/exec_cache.py): the pinned test code's parsed TPC-H/TPC-DS rows
# and DuckDB's results for the pinned transpiled queries. Nothing here depends on a patch.
RUN cd / && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 SKIP_INTEGRATION=0 \
    PYTHONPATH=/opt/sqlglot python /app/exec_cache.py
RUN python -m compileall -q /app \
 && chmod -R a+rX /app

ENV PYTHONDONTWRITEBYTECODE=1 SKIP_INTEGRATION=0
CMD ["python", "-S", "-E", "-s", "/app/main.py"]
