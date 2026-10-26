FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Docker marks the container healthy once `/health` answers.
# The start period leaves time to download the model.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Download the model when the container starts, not when the image is built.
# The image then holds no token, and the same image can serve any model.
CMD ["sh", "-c", "python -m ci_cd.download_model && exec uvicorn ci_cd.api:app --host 0.0.0.0 --port 8000"]
