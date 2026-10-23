# Real-time web app: serves the webcam page and forwards frames to the API.
# Small image - no ML libraries needed here.
FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart httpx

WORKDIR /app
COPY app/ app/

ENV HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    API_URL=http://api:8000

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/', timeout=4)"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8501"]
