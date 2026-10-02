FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip check

COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=12 \
    CMD python -c "import json,urllib.request,sys; \
r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); \
sys.exit(0 if r.status==200 and json.loads(r.read()).get('status')=='healthy' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
