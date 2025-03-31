FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache to a writable directory
ENV TRANSFORMERS_CACHE=/app/cache

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port and run FastAPI app
CMD ["uvicorn", "app.semantic_api:app", "--host", "0.0.0.0", "--port", "7860"]