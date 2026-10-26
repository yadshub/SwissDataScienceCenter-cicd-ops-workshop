FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app/src

# Last, so a new commit SHA doesn't invalidate the cached layers above.
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT

CMD ["python", "-m", "ci_cd.train"]
