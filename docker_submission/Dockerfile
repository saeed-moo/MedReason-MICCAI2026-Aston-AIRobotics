FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDREASON_INPUT_DIR=/input \
    MEDREASON_OUTPUT_DIR=/output \
    MEDREASON_OUTPUT_FILE=/output/results.json \
    MEDREASON_SYSTEM=custom

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY . /opt/app
RUN chmod +x /opt/app/process.py

ENTRYPOINT ["python", "/opt/app/process.py"]
