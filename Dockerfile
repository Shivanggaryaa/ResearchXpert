# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS, PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose dynamic port for Render
EXPOSE ${PORT:-10000}

# Run the app with Gunicorn using Render's PORT
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]
