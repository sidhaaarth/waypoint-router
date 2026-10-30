# Waypoint for your own network: sign-up, login, onboarding, dashboard, live chat, docs and the router API, in one image.
# No demo or showcase content: you bring your own rules and conversations (Teach your router), kept in the /data volume.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DB_PATH=/data/router.db MEMORY_DIR=/data/memory PORT=8888 SELF_HOSTED=1
WORKDIR /srv
COPY requirements.txt .
RUN grep -v '^pytest' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt \
 && useradd -r -u 1000 router && mkdir /data && chown router /data
COPY app/ app/
COPY static/ static/
RUN rm -f static/replay.html static/site/home.html static/site/how.html static/site/results.html static/site/models.html
USER router
VOLUME /data
EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8888') + '/healthz', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8888}"]
