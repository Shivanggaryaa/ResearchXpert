# ───── Smart-Paper-Explainer │ Python 3.11 ─────

# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
E
EXPOSE 10000

# Run the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
