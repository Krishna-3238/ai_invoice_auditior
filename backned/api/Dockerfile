FROM python:3.11.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY backned/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backned/api ./backned/api
COPY backned/config ./backned/config
COPY backned/data/ERP_mockdata ./backned/data/ERP_mockdata
RUN useradd --create-home --uid 10002 erpuser && chown -R erpuser:erpuser /app
USER erpuser
EXPOSE 8000
CMD ["uvicorn", "backned.api.erp_mock_api:app", "--host", "0.0.0.0", "--port", "8000"]
