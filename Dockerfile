# Blast Radius submission: a static test selector.
# The digest is the image the organisers labelled in (keep it: its Python decides test behaviour).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# app/  the selector (stdlib only)
# maps/ built offline from sqlglot @ eb11aec4be8cb2122afe713a4816662c3f0fb2fa by
#       tools/make_maps.sh (per-test coverage, import contexts, class usage, data files,
#       collection-time imports). See CHANGELOG.md.
COPY app/ /app/
COPY maps/ /app/maps/
RUN python -m compileall -q /app \
 && chmod -R a+rX /app

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-S", "-E", "-s", "/app/main.py"]
