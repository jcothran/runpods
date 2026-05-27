# Start with RunPod's official optimized PyTorch base image
FROM runpod/pytorch:2.1.2-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install Ultralytics and the RunPod Serverless SDK
RUN pip install --no-cache-dir ultralytics runpod

# Set up our working directory inside the container
WORKDIR /app

# Copy only your text scripts into the image
COPY handler.py /app/handler.py

# Tell the container to launch your handler script immediately on boot
CMD [ "python", "-u", "/app/handler.py" ]