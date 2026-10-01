FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install system runtime libraries required by OpenCV, FFmpeg, and compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA wheels (or fallback cleanly if on CPU)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create non-root user or persist workspace
EXPOSE 8888 8000

# Default CMD runs Jupyter for development, overrideable to uvicorn later
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]