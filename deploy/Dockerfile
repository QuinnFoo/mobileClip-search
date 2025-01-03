FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

# Create necessary directories
RUN mkdir -p images cache checkpoints

# Copy application files
COPY requirements.txt .

# Install Python dependencies and clean cache
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/apple/ml-mobileclip.git && \
    rm -rf /root/.cache/pip

# Copy other application files
COPY app.py .
COPY templates templates/
COPY checkpoints checkpoints/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Set directory permissions
RUN chmod -R 777 images cache

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120", "--workers", "1", "--log-level", "debug"] 