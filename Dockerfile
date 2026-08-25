# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS dependencies

WORKDIR /app

RUN set -u; \
    for attempt in 1 2 3 4 5; do \
        if apt-get -o Acquire::Retries=5 update \
            && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
                build-essential \
                gcc \
                libpq-dev; then \
            rm -rf /var/lib/apt/lists/*; \
            exit 0; \
        fi; \
        if [ "$attempt" -eq 5 ]; then exit 1; fi; \
        sleep $((attempt * 2)); \
    done

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

FROM python:3.12-slim AS production

WORKDIR /app

COPY --from=dependencies /usr/local /usr/local
COPY requirements-ci.txt ./
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
