# Use multi-platform base image
FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies required for PyAudio and other packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with retries and increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt || \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt || \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PICOVOICE_ACCESS_KEY=""

# Create a script to start the application
RUN echo '#!/bin/bash\n\
PORT="${PORT:-8000}"\n\
exec uvicorn src.api.api:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 300 --ws websockets\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
