import os
import ssl
import urllib.request
import runpod
from ultralytics.models.sam import SAM3SemanticPredictor

#MODEL_PATH = "/tmp/sam3.pt"
MODEL_PATH = "/runpods-volume/sam/sam3.pt"
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
    device=0,   # 👈 FORCE GPU USAGE (0 is the first GPU)    
    half=True,
    save=False, # Turning off save to keep execution times under a second
)
predictor = SAM3SemanticPredictor(overrides=overrides)



import torch
import base64
import cv2
import numpy as np
import requests

def handler(job):
    """The serverless API entry point handling single or batched camera triggers."""
    job_input = job.get('input', {})
    
    # Standardize input to handle either a single image or an explicit batch array
    batch_items = job_input.get("batch_items", [])
    global_exemplar_b64 = job_input.get("exemplar_image_b64", None)
    
    # Global fallback configuration options
    global_return_annotated_image = job_input.get('return_annotated_image', False)
    global_return_boxes = job_input.get('return_boxes', True)
    global_return_polygons = job_input.get('return_polygons', False)

    # Convert single image payload variant into standard batch format under the hood if necessary
    if not batch_items:
        single_url = job_input.get("image_path")
        if single_url:
            batch_items = [{
                "image_path": single_url,
                "prompts": job_input.get("prompts", ["water", "dock", "shoreline"]),
                "source_filename": single_url.split('/')[-1].split('?')[0]
            }]
        else:
            return {"status": "error", "message": "Missing 'batch_items' or 'image_path' in payload."}

    compiled_batch_results = []

    # Iterate over every asset sequentially inside GPU execution memory scope
    for item in batch_items:
        image_url = item.get("image_path")
        text_prompts = item.get("prompts", ["water", "dock", "shoreline"])
        source_filename = item.get("source_filename", "unknown_source")
        
        # Pull item-specific overrides if provided, otherwise default to global switches
        return_annotated_image = item.get('return_annotated_image', global_return_annotated_image)
        return_boxes = item.get('return_boxes', global_return_boxes)
        return_polygons = item.get('return_polygons', global_return_polygons)
        
        # Determine specific item exemplar string
        exemplar_b64 = item.get("exemplar_image_b64", global_exemplar_b64)

        if not image_url:
            compiled_batch_results.append({
                "image_path": "missing",
                "source_filename": source_filename,
                "status": "error",
                "message": "Missing image_path for this specific batch element item."
            })
            continue

        try:
            # 1. Fetch and decode the main scene image
            response = requests.get(image_url)
            image_bytes = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Downloaded image payload failed to decode via OpenCV.")
                
            img_h, img_w = image.shape[:2]
            
            # 2. Setup tracking mode flags
            exemplar_img = None
            exemplar_mask = None
            mode_label = "Text Prompt"

            # Check if an image exemplar string was sent
            if exemplar_b64 and exemplar_b64.strip():
                try:
                    patch_bytes = base64.b64decode(exemplar_b64)
                    patch_arr = np.frombuffer(patch_bytes, np.uint8)
                    exemplar_rgba = cv2.imdecode(patch_arr, cv2.IMREAD_UNCHANGED)
                    
                    if exemplar_rgba is not None:
                        if exemplar_rgba.shape[2] == 4:
                            # Extract BGR channels and use Alpha channel as the isolated binary mask
                            exemplar_img = cv2.cvtColor(exemplar_rgba, cv2.COLOR_BGRA2BGR)
                            alpha_channel = exemplar_rgba[:, :, 3]
                            _, exemplar_mask = cv2.threshold(alpha_channel, 0, 1, cv2.THRESH_BINARY)
                        else:
                            # No alpha transparency channel found, use as solid square crop
                            exemplar_img = exemplar_rgba
                        mode_label = "Visual Exemplar"
                except Exception as b64_err:
                    print(f"⚠️ Failed parsing exemplar string, falling back to text prompts: {b64_err}")

            # 3. Fire the pre-warmed SAM 3 engine based on active mode choice
            predictor.set_image(image_url)
            
            if exemplar_img is not None:
                results = predictor(exemplar=exemplar_img, exemplar_mask=exemplar_mask)
            else:
                results = predictor(text=text_prompts)
            
            # 4. Extract raw bounding box metrics
            raw_boxes = []
            confidences = []
            class_ids = []
            mask_indices = []
            
            if results and results[0].boxes:
                for idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                    class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                    
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                    confidences.append(conf)
                    class_ids.append(class_id)
                    mask_indices.append(idx)

            # 5. Apply NMS to merge overlapping predictions
            indices = []
            if raw_boxes:
                indices = cv2.dnn.NMSBoxes(bboxes=raw_boxes, scores=confidences, score_threshold=0.50, nms_threshold=0.25)
            if len(indices) > 0:
                indices = np.array(indices).flatten()

            # 6. Build clean telemetry dictionary array
            detected_objects = []
            final_kept_indices = []
            has_masks = hasattr(results[0], 'masks') and results[0].masks is not None

            for i in indices:
                x, y, w, h = raw_boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                conf = confidences[i]
                orig_idx = mask_indices[i]
                
                # Map clean string label based on execution mode
                if exemplar_img is not None:
                    label_name = mode_label
                else:
                    label_name = text_prompts[class_ids[i]] if class_ids[i] < len(text_prompts) else f"Object_{class_ids[i]}"
                
                pixel_area = 0
                polygon_coords = []
                
                if has_masks:
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    
                    pixel_area = int(np.sum(binary_mask))
                    
                    if return_polygons:
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            polygon_coords = largest_contour.reshape(-1, 2).tolist()

                item_data = {"label": label_name, "conf": conf, "pixel_area": pixel_area}
                if return_boxes:
                    item_data["box"] = [x1, y1, x2, y2]
                if return_polygons:
                    item_data["polygon"] = polygon_coords

                detected_objects.append(item_data)
                final_kept_indices.append(orig_idx)

            annotated_image_b64 = None

            # 7. Core OpenCV Drawing Layer
            if return_annotated_image and detected_objects:
                np.random.seed(42)
                colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
                
                if has_masks:
                    for out_idx, orig_idx in enumerate(final_kept_indices):
                        binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                        if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                            binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        color = colors[out_idx % len(colors)].tolist()
                        colored_mask = np.zeros_like(image, dtype=np.uint8)
                        colored_mask[binary_mask == 1] = color
                        cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

                for out_idx, obj in enumerate(detected_objects):
                    if "box" in obj:
                        x1, y1, x2, y2 = obj["box"]
                    else:
                        idx_in_raw = final_kept_indices[out_idx]
                        x, y, w, h = raw_boxes[idx_in_raw]
                        x1, y1, x2, y2 = x, y, x + w, y + h

                    label = f"{obj['label']} ({obj['pixel_area']:,} px)"
                    color = colors[out_idx % len(colors)].tolist()
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                success, encoded_image = cv2.imencode('.jpg', image)
                if success:
                    annotated_image_b64 = base64.b64encode(encoded_image).decode('utf-8')

            # Append successful frame evaluations back to the batch tracking response array
            compiled_batch_results.append({
                "image_path": image_url,
                "source_filename": source_filename,
                "status": "success",
                "pipeline_mode": mode_label,
                "predictions": detected_objects,
                "annotated_image_b64": annotated_image_b64
            })

        except Exception as frame_err:
            compiled_batch_results.append({
                "image_path": image_url,
                "source_filename": source_filename,
                "status": "error",
                "message": f"Error during processing: {str(frame_err)}"
            })

        # Explicitly empty out cached VRAM garbage between loop calculations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "status": "success",
        "batch_results": compiled_batch_results
    }

runpod.serverless.start({"handler": handler})

'''
import base64
import cv2
import numpy as np
import requests

def handler(job):
    """The serverless API entry point called on every new camera image trigger."""
    job_input = job.get('input', {})
    image_url = job_input.get("image_path")
    
    # Optional Exemplar Input
    exemplar_b64 = job_input.get("exemplar_image_b64", None)
    
    # Standard Text Fallbacks
    text_prompts = job_input.get("prompts", ["water", "dock", "shoreline"])
    
    # Configuration Toggles
    return_annotated_image = job_input.get('return_annotated_image', False)
    return_boxes = job_input.get('return_boxes', True)
    return_polygons = job_input.get('return_polygons', False)

    if not image_url:
        return {"status": "error", "message": "Missing 'image_path' in request payload."}

    try:
        # 1. Fetch and decode the main scene image
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]
        
        # 2. Setup tracking mode flags
        exemplar_img = None
        exemplar_mask = None
        mode_label = "Text Prompt"

        # Check if an image exemplar string was actually sent
        if exemplar_b64 and exemplar_b64.strip():
            try:
                patch_bytes = base64.b64decode(exemplar_b64)
                patch_arr = np.frombuffer(patch_bytes, np.uint8)
                # IMREAD_UNCHANGED preserves the 4th Alpha Transparency Channel
                exemplar_rgba = cv2.imdecode(patch_arr, cv2.IMREAD_UNCHANGED)
                
                if exemplar_rgba is not None:
                    if exemplar_rgba.shape[2] == 4:
                        # Extract BGR channels and use Alpha channel as the isolated binary mask
                        exemplar_img = cv2.cvtColor(exemplar_rgba, cv2.COLOR_BGRA2BGR)
                        alpha_channel = exemplar_rgba[:, :, 3]
                        _, exemplar_mask = cv2.threshold(alpha_channel, 0, 1, cv2.THRESH_BINARY)
                    else:
                        # No alpha transparency channel found, use as a solid square crop
                        exemplar_img = exemplar_rgba
                    mode_label = "Visual Exemplar"
            except Exception as b64_err:
                print(f"⚠️ Failed parsing exemplar string, falling back to text prompts: {b64_err}")

        # 3. Fire the pre-warmed SAM 3 engine based on active mode choice
        predictor.set_image(image_url)
        
        if exemplar_img is not None:
            # Concept Search Mode
            results = predictor(exemplar=exemplar_img, exemplar_mask=exemplar_mask)
        else:
            # Standard Keyword Search Mode
            results = predictor(text=text_prompts)
        
        # 4. Extract raw bounding box metrics
        raw_boxes = []
        confidences = []
        class_ids = []
        mask_indices = []
        
        if results and results[0].boxes:
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                raw_boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                confidences.append(conf)
                class_ids.append(class_id)
                mask_indices.append(idx)

        # 5. Apply NMS to merge overlapping predictions
        indices = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(bboxes=raw_boxes, scores=confidences, score_threshold=0.50, nms_threshold=0.25)
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # 6. Build clean telemetry dictionary array
        detected_objects = []
        final_kept_indices = []
        has_masks = hasattr(results[0], 'masks') and results[0].masks is not None

        for i in indices:
            x, y, w, h = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            conf = confidences[i]
            orig_idx = mask_indices[i]
            
            # Map clean string label based on execution mode
            if exemplar_img is not None:
                label_name = mode_label
            else:
                label_name = text_prompts[class_ids[i]] if class_ids[i] < len(text_prompts) else f"Object_{class_ids[i]}"
            
            pixel_area = 0
            polygon_coords = []
            
            if has_masks:
                binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                pixel_area = int(np.sum(binary_mask))
                
                if return_polygons:
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon_coords = largest_contour.reshape(-1, 2).tolist()

            item_data = {"label": label_name, "conf": conf, "pixel_area": pixel_area}
            if return_boxes:
                item_data["box"] = [x1, y1, x2, y2]
            if return_polygons:
                item_data["polygon"] = polygon_coords

            detected_objects.append(item_data)
            final_kept_indices.append(orig_idx)

        return_output = {
            "status": "success",
            "pipeline_mode": mode_label,
            "predictions": detected_objects,
            "annotated_image_b64": None,
            "message": f"Successfully processed image frame using {mode_label} mode."
        }

        # 7. Core OpenCV Drawing Layer
        if return_annotated_image and detected_objects:
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
            
            if has_masks:
                for out_idx, orig_idx in enumerate(final_kept_indices):
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    color = colors[out_idx % len(colors)].tolist()
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[binary_mask == 1] = color
                    cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

            for out_idx, obj in enumerate(detected_objects):
                if "box" in obj:
                    x1, y1, x2, y2 = obj["box"]
                else:
                    idx_in_raw = final_kept_indices[out_idx]
                    x, y, w, h = raw_boxes[idx_in_raw]
                    x1, y1, x2, y2 = x, y, x + w, y + h

                label = f"{obj['label']} ({obj['pixel_area']:,} px)"
                color = colors[out_idx % len(colors)].tolist()
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                return_output["annotated_image_b64"] = base64.b64encode(encoded_image).decode('utf-8')

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''

'''
import base64
import cv2
import numpy as np
import requests

def handler(job):
    """The serverless API entry point called on every new camera image trigger."""
    job_input = job.get('input', {})
    image_url = job_input.get("image_path")
    text_prompts = job_input.get("prompts", ["person", "bus", "glasses"])
    
    # Payload Toggles
    return_annotated_image = job_input.get('return_annotated_image', False)
    return_boxes = job_input.get('return_boxes', True)          # Default to True
    return_polygons = job_input.get('return_polygons', False)    # Default to False for performance

    if not image_url:
        return {"status": "error", "message": "Missing 'image_path' in request payload."}

    try:
        # 1. Fetch and decode raw image bytes
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]
        
        # 2. Run inference using the pre-warmed model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        raw_boxes = []
        confidences = []
        class_ids = []
        mask_indices = []
        
        # 3. Harvest raw predictions
        if results and results[0].boxes:
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                raw_boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                confidences.append(conf)
                class_ids.append(class_id)
                mask_indices.append(idx)

        # 4. Apply Non-Maximum Suppression (NMS) to clear overlapping duplicates
        indices = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(
                bboxes=raw_boxes, 
                scores=confidences, 
                score_threshold=0.25, 
                nms_threshold=0.45
            )
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # 5. Compile the clean, filtered object collection based on requested flags
        detected_objects = []
        final_kept_indices = []
        has_masks = hasattr(results[0], 'masks') and results[0].masks is not None

        for i in indices:
            x, y, w, h = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            class_id = class_ids[i]
            conf = confidences[i]
            orig_idx = mask_indices[i]
            
            label_name = text_prompts[class_id] if class_id < len(text_prompts) else f"Object_{class_id}"
            
            pixel_area = 0
            polygon_coords = []
            
            if has_masks:
                binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Always calculate pixel area for analytics
                pixel_area = int(np.sum(binary_mask))
                
                # Only calculate heavy vector contours if the client explicitly wants them
                if return_polygons:
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon_coords = largest_contour.reshape(-1, 2).tolist()

            # Build item dictionary dynamically based on toggles
            item_data = {
                "label": label_name,
                "conf": conf,
                "pixel_area": pixel_area
            }
            
            if return_boxes:
                item_data["box"] = [x1, y1, x2, y2]
            if return_polygons:
                item_data["polygon"] = polygon_coords

            detected_objects.append(item_data)
            final_kept_indices.append(orig_idx)

        # 6. Base JSON payload response structure
        return_output = {
            "status": "success",
            "predictions": detected_objects,
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 7. Blend masks and draw boxes onto image if requested
        if return_annotated_image and detected_objects:
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
            
            if has_masks:
                for out_idx, orig_idx in enumerate(final_kept_indices):
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    
                    color = colors[out_idx % len(colors)].tolist()
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[binary_mask == 1] = color
                    cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

            for out_idx, obj in enumerate(detected_objects):
                # Fallback to local coordinates calculation if box wasn't requested in text output
                if "box" in obj:
                    x1, y1, x2, y2 = obj["box"]
                else:
                    # Quick reconstruction for drawing layers
                    idx_in_raw = final_kept_indices[out_idx]
                    x, y, w, h = raw_boxes[idx_in_raw]
                    x1, y1, x2, y2 = x, y, x + w, y + h

                label = f"{obj['label']} ({obj['pixel_area']:,} px)"
                color = colors[out_idx % len(colors)].tolist()
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2, cv2.LINE_AA)
            
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                return_output["annotated_image_b64"] = base64.b64encode(encoded_image).decode('utf-8')

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''
        
'''
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
        return {"status": "error", "message": "Missing 'image_path' in request payload."}

    try:
        # 1. Fetch and decode raw image bytes for canvas operations
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]
        
        # 2. Run inference using the pre-warmed model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        raw_boxes = []
        confidences = []
        class_ids = []
        mask_indices = []
        
        # 3. Harvest raw predictions
        if results and results[0].boxes:
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                raw_boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                confidences.append(conf)
                class_ids.append(class_id)
                mask_indices.append(idx)

        # 4. Apply Non-Maximum Suppression (NMS) to clear overlapping duplicates
        indices = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(
                bboxes=raw_boxes, 
                scores=confidences, 
                score_threshold=0.45, 
                nms_threshold=0.25
            )
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # 5. Compile the clean, filtered object collection with area & polygon metrics
        detected_objects = []
        final_kept_indices = []
        has_masks = hasattr(results[0], 'masks') and results[0].masks is not None

        for i in indices:
            x, y, w, h = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            class_id = class_ids[i]
            conf = confidences[i]
            orig_idx = mask_indices[i]
            
            label_name = text_prompts[class_id] if class_id < len(text_prompts) else f"Object_{class_id}"
            
            pixel_area = 0
            polygon_coords = []
            
            # Extract analytics from the binary pixel array if masks exist
            if has_masks:
                binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                
                # Match native image canvas dimensions if working with downscaled tensors
                if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Metric A: Total surface area calculated via pixel count summation
                pixel_area = int(np.sum(binary_mask))
                
                # Metric B: Vectorized polygon contour generation
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon_coords = largest_contour.reshape(-1, 2).tolist()

            detected_objects.append({
                "box": [x1, y1, x2, y2],
                "label": label_name,
                "conf": conf,
                "pixel_area": pixel_area,
                "polygon": polygon_coords
            })
            final_kept_indices.append(orig_idx)

        # 6. Base JSON text payload response structure
        return_output = {
            "status": "success",
            "predictions": detected_objects,
            "detected_boxes": [obj["box"] for obj in detected_objects],
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 7. Blend masks and draw boxes onto image if requested
        if return_annotated_image and detected_objects:
            np.random.seed(42)  # Maintain stable color palette across runs
            colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
            
            # Loop 1: Layer all translucent masks first
            if has_masks:
                for out_idx, orig_idx in enumerate(final_kept_indices):
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    
                    color = colors[out_idx % len(colors)].tolist()
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[binary_mask == 1] = color
                    
                    # Blend colored layer into working image array with 40% mask transparency opacity
                    cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

            # Loop 2: Layer crisp bounding borders and text elements on top
            for out_idx, obj in enumerate(detected_objects):
                x1, y1, x2, y2 = obj["box"]
                label = f"{obj['label']} ({obj['pixel_area']:,} px)"
                color = colors[out_idx % len(colors)].tolist()
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2, cv2.LINE_AA)
            
            # Encode modified image array matrix to Base64 ASCII string layout
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                return_output["annotated_image_b64"] = base64.b64encode(encoded_image).decode('utf-8')

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''
        
'''
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
        # 1. Fetch and decode raw image bytes
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # 2. Run inference using pre-warmed SAM 3 model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        raw_boxes = []
        confidences = []
        class_ids = []
        mask_indices = []  # Keep track of which mask belongs to which box
        
        # 3. Harvest raw predictions (boxes and masks)
        if results and results[0].boxes:
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                w = x2 - x1
                h = y2 - y1
                
                raw_boxes.append([x1, y1, w, h])
                confidences.append(conf)
                class_ids.append(class_id)
                mask_indices.append(idx)

        # 4. Apply Non-Maximum Suppression (NMS) to eliminate overlaps
        indices = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(
                bboxes=raw_boxes, 
                scores=confidences, 
                score_threshold=0.25, 
                nms_threshold=0.45
            )
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # 5. Compile the final clean filtered object collection
        detected_objects = []
        final_kept_indices = [] # Track the original indices that survived NMS for masks
        
        for i in indices:
            x, y, w, h = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            class_id = class_ids[i]
            conf = confidences[i]
            
            if class_id < len(text_prompts):
                label_name = text_prompts[class_id]
            else:
                label_name = f"Object_{class_id}"
                
            detected_objects.append({
                "box": [x1, y1, x2, y2],
                "label": label_name,
                "conf": conf
            })
            final_kept_indices.append(mask_indices[i])

        # 6. Construct response payload
        return_output = {
            "status": "success",
            "predictions": detected_objects,
            "detected_boxes": [obj["box"] for obj in detected_objects],
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 7. Draw the clean masks and boxes if requested
        if return_annotated_image and detected_objects:
            # Generate distinct colors for different classes/objects using a seedable colormap
            # We'll generate up to 20 random color bounds to keep things vibrant
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
            
            # Draw Masks First (so text labels/bounding box borders remain crisp on top)
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                # Get native image dimensions to resize masks if needed
                img_h, img_w = image.shape[:2]
                
                for out_idx, orig_idx in enumerate(final_kept_indices):
                    # Extract the specific binary mask data matrix (0s and 1s)
                    # Convert torch tensor back to a standard numpy matrix array
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    
                    # Ensure mask matches exact OpenCV frame canvas dimensions
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Choose color based on the item index
                    color = colors[out_idx % len(colors)].tolist()
                    
                    # Create a solid color canvas layer matching the original image size
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[binary_mask == 1] = color
                    
                    # Blend the colored mask layer transparently onto our working image frame
                    # Alpha=0.6 (keep original), Beta=0.4 (apply tinted mask transparency mix)
                    cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

            # Draw Bounding Boxes and Text Labels Second
            for out_idx, obj in enumerate(detected_objects):
                x1, y1, x2, y2 = obj["box"]
                label = f"{obj['label']} {obj['conf']:.2f}"
                
                # Match the box border color to the mask color we generated above
                color = colors[out_idx % len(colors)].tolist()
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, color, 2, cv2.LINE_AA)
            
            # Encode modified image array matrix back into JPEG memory buffer
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                b64_string = base64.b64encode(encoded_image).decode('utf-8')
                return_output["annotated_image_b64"] = b64_string

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''

'''
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
        # 1. Fetch and decode raw image bytes
        response = requests.get(image_url)
        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # 2. Run inference using pre-warmed model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        raw_boxes = []
        confidences = []
        class_ids = []
        
        # 3. Harvest raw predictions
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                # OpenCV NMS requires bounding boxes in [x, y, width, height] format
                w = x2 - x1
                h = y2 - y1
                
                raw_boxes.append([x1, y1, w, h])
                confidences.append(conf)
                class_ids.append(class_id)

        # 4. Apply Non-Maximum Suppression (NMS) to eliminate overlaps
        # iou_threshold=0.45: Boxes overlapping more than 45% with a higher-conf box are suppressed
        # score_threshold=0.25: Discard detections lower than 25% confidence
        indices = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(
                bboxes=raw_boxes, 
                scores=confidences, 
                score_threshold=0.50, #was 0.25
                nms_threshold=0.25 #was 0.45
            )
        
        # Flatten indices array if necessary (handles differences across OpenCV versions)
        if len(indices) > 0:
            indices = np.array(indices).flatten()

        # 5. Compile the final clean filtered object collection
        detected_objects = []
        for i in indices:
            x, y, w, h = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            class_id = class_ids[i]
            conf = confidences[i]
            
            if class_id < len(text_prompts):
                label_name = text_prompts[class_id]
            else:
                label_name = f"Object_{class_id}"
                
            detected_objects.append({
                "box": [x1, y1, x2, y2],
                "label": label_name,
                "conf": conf
            })

        # 6. Construct response payload
        return_output = {
            "status": "success",
            "predictions": detected_objects,
            "detected_boxes": [obj["box"] for obj in detected_objects],
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 7. Draw the clean, non-overlapping boxes if requested
        if return_annotated_image and detected_objects:
            for obj in detected_objects:
                x1, y1, x2, y2 = obj["box"]
                label = f"{obj['label']} {obj['conf']:.2f}"
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                b64_string = base64.b64encode(encoded_image).decode('utf-8')
                return_output["annotated_image_b64"] = b64_string

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''

'''
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
        
        # 2. Run inference using the pre-warmed model
        predictor.set_image(image_url)
        results = predictor(text=text_prompts)
        
        detected_objects = []
        
        # 3. Match your initial logic checking if boxes exist
        if results and results[0].boxes:
            boxes = results[0].boxes
            
            for box in boxes:
                # Extract coordinates as standard integers for OpenCV compatibility
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Safely extract confidence matrix strings
                conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                
                # Map the label directly to your text prompt string to avoid missing attribute errors
                if class_id < len(text_prompts):
                    label_name = text_prompts[class_id]
                else:
                    label_name = f"Object_{class_id}"
                
                # Append data formatted cleanly for your local client tracking pipeline
                detected_objects.append({
                    "box": [x1, y1, x2, y2],
                    "label": label_name,
                    "conf": conf
                })

        # 4. Construct response dictionary matching your original layout
        return_output = {
            "status": "success",
            "predictions": detected_objects,       # Local text logging pipeline reads this
            "detected_boxes": [obj["box"] for obj in detected_objects], # Keeps your original exact coordinate array layout intact
            "annotated_image_b64": None,
            "message": f"Successfully processed {image_url}"
        }

        # 5. Draw bounding boxes on demand if the local client asks for it
        if return_annotated_image and detected_objects:
            for obj in detected_objects:
                x1, y1, x2, y2 = obj["box"]
                label = f"{obj['label']} {obj['conf']:.2f}"
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                b64_string = base64.b64encode(encoded_image).decode('utf-8')
                return_output["annotated_image_b64"] = b64_string

        return return_output

    except Exception as e:
        return {"status": "error", "message": str(e)}
'''

#import base64
#import io
#import cv2
#import numpy as np
#import requests

'''
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
