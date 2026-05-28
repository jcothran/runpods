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

#import base64
#import io
#import cv2
#import numpy as np
#import requests

import base64
import cv2
import numpy as np
import requests

def handler(job):
    """The serverless API entry point called on every new camera image trigger."""
    job_input = job.get('input', {})
    image_url = job_input.get("image_path")
    text_prompts = job_input.get("prompts", ["person", "bus", "glasses"])
    return_annotated_image = job_input.get('return_annotated_image', False)
    
    if not image_url:
        return {"error": "Missing 'image_path' in request payload."}

    try:
        # 1. Fetch and decode the raw image bytes for OpenCV annotation drawing
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # 2. Run inference using the pre-warmed SAM 3 / Ultralytics model
        # Note: If your model requires loading an image array rather than a URL string, 
        # you can pass 'image' directly instead: predictor.set_image(image)
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        # 3. Parse real prediction tracking elements out of the model outputs object
        detected_objects = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            
            # Loop over every item found by the model
            for box in boxes:
                # Extract coordinates as standard integers for OpenCV compatibility
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Extract confidence score and class ID metadata
                conf = float(box.conf[0].item()) if box.conf is not None else 1.0
                class_id = int(box.cls[0].item()) if box.cls is not None else 0
                
                # Get the string label name (fallback to ID if names map is missing)
                label_name = predictor.names.get(class_id, f"object_{class_id}")
                
                # Build the exact dictionary structure expected by your local client pipeline
                detected_objects.append({
                    "box": [x1, y1, x2, y2],
                    "label": label_name,
                    "conf": conf
                })

        # 4. Construct standard JSON text response payload
        return_output = {
            "status": "success",
            "predictions": detected_objects,
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 5. If requested, draw boxes onto the downloaded frame and encode to base64
        if return_annotated_image and detected_objects:
            for obj in detected_objects:
                x1, y1, x2, y2 = obj["box"]
                label = f"{obj['label']} {obj['conf']:.2f}"
                
                # Draw the bounding box rectangle (Green outline, thickness of 3 pixels)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add text label above the rectangle boundary
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the modified image matrix back into a JPEG memory buffer
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                # Convert the byte buffer into a clean Base64 string for the JSON payload
                b64_string = base64.b64encode(encoded_image).decode('utf-8')
                return_output["annotated_image_b64"] = b64_string

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}

'''
def handler(job):
    job_input = job.get('input', {})
    image_path = job_input.get('image_path')
    prompts = job_input.get('prompts', [])
    
    # New configuration flag (defaults to False if not passed)
    return_annotated_image = job_input.get('return_annotated_image', False)

    # 1. Fetch and decode the image from the WebCOOS S3 URL
    response = requests.get(image_path)
    image_bytes = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    
    # 2. RUN YOUR SAM 3 / ULTRALYTICS INFERENCE HERE
    # Let's assume your model outputs a list of bounding boxes: [[x1, y1, x2, y2, label, confidence], ...]
    # For demonstration, let's say 'detected_objects' holds your parsed results:
    detected_objects = [
        {"box": [100, 150, 400, 500], "label": "bus", "conf": 0.92},
        {"box": [120, 200, 180, 300], "label": "person", "conf": 0.88}
    ]

    # 3. Standard text response data dictionary
    return_output = {
        "predictions": detected_objects,
        "annotated_image_b64": None
    }

    # 4. If the user requested the visual image layout, draw the boxes
    if return_annotated_image:
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["box"]
            label = f"{obj['label']} {obj['conf']:.2f}"
            
            # Draw the bounding box rectangle (Green: 0, 255, 0; Thickness: 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add text label above the rectangle boundary
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Encode the modified image array matrix back into a JPEG memory buffer
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            # Convert memory buffer to a clean Base64 ASCII string layout
            b64_string = base64.b64encode(encoded_image).decode('utf-8')
            return_output["annotated_image_b64"] = b64_string

    return return_output
'''

'''
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
'''

# Register with RunPod Serverless
runpod.serverless.start({"handler": handler})
