#FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Ensure Ultralytics dependencies are fully pre-baked into the image layer
RUN pip install --no-cache-dir \
    timm==1.0.27 \
    typer==0.25.1 \
    safetensors==0.7.0 \
    huggingface-hub==1.18.0 \
    hf-xet==1.5.0
    
# 3. Set up your app
WORKDIR /app
COPY handler.py /app/handler.py

# 4. Run with unbuffered logging active
CMD [ "python", "-u", "/app/handler.py" ]
