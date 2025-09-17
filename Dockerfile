# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF & FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Run Flask app (Hugging Face expects process on port 7860)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--workers=1", "--threads=2", "--timeout=120"]
