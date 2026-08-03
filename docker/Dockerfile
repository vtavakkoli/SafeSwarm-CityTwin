FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-dev.txt

COPY . .

CMD ["python", "experiments/run_city_benchmark.py"]
