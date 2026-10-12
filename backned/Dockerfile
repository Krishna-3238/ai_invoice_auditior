FROM python:3.11.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-deu poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backned ./backned
COPY fronted ./fronted
RUN useradd --create-home --uid 10001 auditor && mkdir -p /app/outputs/reports /app/logs && chown -R auditor:auditor /app
USER auditor
EXPOSE 8000
CMD ["uvicorn", "backned.api.agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
