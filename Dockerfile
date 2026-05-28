FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 1. Install the essential Linux graphics rendering pipes OpenCV needs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install your python libraries
RUN pip install --no-cache-dir ultralytics runpod

# 3. Set up your app
WORKDIR /app
COPY handler.py /app/handler.py

# 4. Run with unbuffered logging active
CMD [ "python", "-u", "/app/handler.py" ]
