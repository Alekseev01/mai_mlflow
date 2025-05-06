FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training /app/training

RUN touch /app/healthcheck

CMD ["python", "training/train.py"]