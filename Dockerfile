# MedAI Radiologia - Production Docker Container
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdcm-tools \
    dcmtk \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY templates/ ./templates/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /app/uploads /var/log/medai

# Set permissions
RUN chmod +x src/*.py

# Create non-root user for security
RUN useradd -m -u 1000 medai && \
    chown -R medai:medai /app /var/log/medai
USER medai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8084 11112

# Default command
CMD ["python", "src/web_server.py", "--config", "config/production.json"]
