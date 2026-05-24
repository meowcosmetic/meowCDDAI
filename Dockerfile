FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

WORKDIR /app

# Install Python and system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for PIP in Ubuntu 24.04
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Fix pip for python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy requirements first for better caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8102

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=main.py
ENV YOLO_CONFIG_DIR=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib

# Run the application
CMD ["python", "main.py"]
