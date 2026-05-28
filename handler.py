import os
import ssl
import urllib.request
import runpod
from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "/tmp/sam3.pt"
MODEL_URL = "http://floridaapdata.org/sam3.pt"

def download_model_if_needed():
    """Checks if the 4GB SAM3 file is present; if not, downloads it, bypassing SSL redirects."""
    if not os.path.exists(MODEL_PATH):
        print(f"📥 SAM3 weights missing. Downloading from {MODEL_URL}...")
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # 2. Create the python equivalent to 'wget --no-check-certificate'
        # This forces Python to ignore SSL errors when the server redirects to HTTPS
        insecure_context = ssl._create_unverified_context()
        
        # 3. Build an opener that uses this insecure context and install it globally
        handler = urllib.request.HTTPSHandler(context=insecure_context)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)
        
        # 4. Perform the download
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Download complete!")
    else:
        print("🧠 SAM3 weights already cached in container memory.")

# 1. Trigger the download immediately when the container spins up
download_model_if_needed()

# 2. Warm up the predictor globally so it sits ready in VRAM
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    half=True,
    save=False, # Turning off save to keep execution times under a second
)
predictor = SAM3SemanticPredictor(overrides=overrides)


def handler(job):
    """The serverless API entry point called on every new camera image trigger."""
    job_input = job['input']
    image_url = job_input.get("image_path")
    text_prompts = job_input.get("prompts", ["person", "bus", "glasses"])
    
    if not image_url:
        return {"error": "Missing 'image_path' in request payload."}

    try:
        # Run inference using the pre-warmed GPU model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        # Extract metadata out of the results object (e.g., box coordinates, classes)
        # You can adjust what you return based on your downstream web app needs
        boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
        
        return {
            "status": "success",
            "detected_boxes": boxes,
            "message": f"Successfully processed {image_url}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Register with RunPod Serverless
runpod.serverless.start({"handler": handler})
