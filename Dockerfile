FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir opencv-python-headless numpy

CMD ["python", "Dimensiomittaus_tiedostolle.py"]
