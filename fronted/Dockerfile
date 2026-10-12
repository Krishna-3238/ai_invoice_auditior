FROM python:3.11.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY fronted/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY fronted ./fronted
RUN useradd --create-home --uid 10003 streamlit && chown -R streamlit:streamlit /app
USER streamlit
EXPOSE 8501
CMD ["streamlit", "run", "fronted/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
