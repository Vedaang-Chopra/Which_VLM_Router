FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch to keep image smaller (change for GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=configs/artemis.yaml

# Run with uvicorn
# Run with uvicorn (Default Server Mode)
CMD ["uvicorn", "system_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: Run as CLI
# docker run --rm -v $(pwd)/data:/app/data artemis-router python main.py --prompt "Describe this image" --image "data/sample.jpg"

