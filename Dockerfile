FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-api.txt

COPY captcha.py digit_ocr.py ocr_api.py ocr_worker.py ./
COPY tessdata/eng.traineddata ./tessdata/

USER 65532:65532
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "ocr_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
