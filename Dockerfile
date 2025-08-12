FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache
COPY requirements.txt .

# Install dependencies in specific order
RUN pip install --no-cache-dir Werkzeug==2.0.3 && \
    pip install --no-cache-dir Flask==2.0.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=8080

# Command to run the application
CMD exec gunicorn --bind :$PORT app:app --workers 1 --threads 8 --timeout 0
