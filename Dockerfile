#FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# 1. Install git (Required to clone and build the CLIP dependency)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. Install Python libraries and pre-bake ALL Ultralytics/SAM3 dependencies
RUN python -m pip install --no-cache-dir \
    ultralytics \
    runpod \
    timm==1.0.27 \
    typer==0.25.1 \
    safetensors==0.7.0 \
    huggingface-hub==1.18.0 \
    hf-xet==1.5.0 \
    regex==2026.5.9 \
    ftfy==6.3.1 \
    git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d
    
# 3. Set up your app
WORKDIR /app
COPY handler.py /app/handler.py

# 4. Run with unbuffered logging active
CMD [ "python", "-u", "/app/handler.py" ]
