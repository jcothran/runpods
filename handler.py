import os
import urllib.request
import runpod
from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "/runpod-volume/sam/sam3.pt"

REMOTE_TAR_URL = "http://saludasys.org/webcoos.tar"  # 🌐 Source server link destination

# Warm up the predictor globally so it sits ready in VRAM
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    device=0,   # 👈 FORCE GPU USAGE (0 is the first GPU)
    imgsz=1008, #was 1024, gemini suggested 1008 as optimal
    half=True,
    save=False, # Turning off save to keep execution times under a second
)
predictor = SAM3SemanticPredictor(overrides=overrides)


import torch
import base64
import cv2
import tarfile
import io
import numpy as np
import requests

def handler(job):
    job_input = job.get('input', {})
    batch_items = job_input.get("batch_items", [])
    use_tar_url = job_input.get("use_tar_url", False)
    global_exemplar_b64 = job_input.get("exemplar_image_b64", None)
    
    global_return_annotated_image = job_input.get('return_annotated_image', False)
    global_return_boxes = job_input.get('return_boxes', True)
    global_return_polygons = job_input.get('return_polygons', False)

    extracted_dir = "/tmp/extracted_images"
    if os.path.exists(extracted_dir):
        shutil = __import__('shutil')
        shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir, exist_ok=True)

    # Fetch and extract the hosted batch archive file if flag is enabled
    if use_tar_url:
        try:
            print(f"📥 Stream fetching batch bundle from server: {REMOTE_TAR_URL}")
            response = requests.get(REMOTE_TAR_URL, timeout=30, verify=False)
            if response.status_code == 200:
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r") as tar:
                    tar.extractall(path=extracted_dir)
                print(f"🔓 Successfully unpacked hosted tar layout directly to {extracted_dir}")
            else:
                print(f"⚠️ Web Server returned status code: {response.status_code}. Using fallback loop.")
        except Exception as tar_err:
            print(f"❌ Failed processing web hosted tarball attachment routing: {tar_err}")

    if not batch_items:
        single_url = job_input.get("image_path")
        if single_url:
            batch_items = [{
                "image_path": single_url,
                "local_archive_filename": single_url.split('/')[-1].split('?')[0],
                "prompts": job_input.get("prompts", ["water", "dock", "shoreline"]),
                "source_filename": single_url.split('/')[-1].split('?')[0]
            }]
        else:
            return {"status": "error", "message": "Missing execution payload configuration context."}

    compiled_batch_results = []

    for item in batch_items:
        image_url = item.get("image_path")
        local_archive_filename = item.get("local_archive_filename")
        text_prompts = item.get("prompts", ["water", "dock", "shoreline"])
        source_filename = item.get("source_filename", "unknown_source")
        
        return_annotated_image = item.get('return_annotated_image', global_return_annotated_image)
        return_boxes = item.get('return_boxes', global_return_boxes)
        return_polygons = item.get('return_polygons', global_return_polygons)
        exemplar_b64 = item.get("exemplar_image_b64", global_exemplar_b64)

        image = None
        
        # Method A: Try reading the uncompressed image out of local storage cache
        if local_archive_filename:
            target_local_path = os.path.join(extracted_dir, local_archive_filename)
            if os.path.exists(target_local_path):
                image = cv2.imread(target_local_path, cv2.IMREAD_COLOR)

        # Method B: Absolute fallback tracking loop execution
        if image is None and image_url:
            try:
                response = requests.get(image_url, timeout=10)
                image_bytes = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            except Exception as dl_err:
                print(f"⚠️ Remote fallback fetch loop failed for link {image_url}: {dl_err}")

        if image is None:
            compiled_batch_results.append({
                "image_path": image_url,
                "source_filename": source_filename,
                "status": "error",
                "message": "Image layout matrix could not be resolved from archive or web extraction points."
            })
            continue

        try:
            img_h, img_w = image.shape[:2]
            exemplar_img, exemplar_mask = None, None
            mode_label = "Text Prompt"

            if exemplar_b64 and exemplar_b64.strip():
                try:
                    patch_bytes = base64.b64decode(exemplar_b64)
                    patch_arr = np.frombuffer(patch_bytes, np.uint8)
                    exemplar_rgba = cv2.imdecode(patch_arr, cv2.IMREAD_UNCHANGED)
                    if exemplar_rgba is not None:
                        if exemplar_rgba.shape[2] == 4:
                            exemplar_img = cv2.cvtColor(exemplar_rgba, cv2.COLOR_BGRA2BGR)
                            _, exemplar_mask = cv2.threshold(exemplar_rgba[:, :, 3], 0, 1, cv2.THRESH_BINARY)
                        else:
                            exemplar_img = exemplar_rgba
                        mode_label = "Visual Exemplar"
                except Exception as b64_err:
                    print(f"⚠️ Exemplar processing issue: {b64_err}")

            predictor.set_image(image)
            results = predictor(exemplar=exemplar_img, exemplar_mask=exemplar_mask) if exemplar_img is not None else predictor(text=text_prompts)
            
            raw_boxes, confidences, class_ids, mask_indices = [], [], [], []
            if results and results[0].boxes:
                for idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item()) if (hasattr(box, 'conf') and box.conf is not None) else 1.0
                    class_id = int(box.cls[0].item()) if (hasattr(box, 'cls') and box.cls is not None) else 0
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(conf)
                    class_ids.append(class_id)
                    mask_indices.append(idx)

            indices = cv2.dnn.NMSBoxes(bboxes=raw_boxes, scores=confidences, score_threshold=0.50, nms_threshold=0.25) if raw_boxes else []
            if len(indices) > 0:
                indices = np.array(indices).flatten()

            detected_objects, final_kept_indices = [], []
            has_masks = hasattr(results[0], 'masks') and results[0].masks is not None

            for i in indices:
                x, y, w, h = raw_boxes[i]
                orig_idx = mask_indices[i]
                label_name = mode_label if exemplar_img is not None else (text_prompts[class_ids[i]] if class_ids[i] < len(text_prompts) else f"Object_{class_ids[i]}")
                
                pixel_area, polygon_coords = 0, []
                if has_masks:
                    binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                    if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    pixel_area = int(np.sum(binary_mask))
                    if return_polygons:
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            polygon_coords = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()

                item_data = {"label": label_name, "conf": confidences[i], "pixel_area": pixel_area}
                if return_boxes:
                    item_data["box"] = [x, y, x + w, y + h]
                if return_polygons:
                    item_data["polygon"] = polygon_coords
                detected_objects.append(item_data)
                final_kept_indices.append(orig_idx)

            annotated_image_b64 = None
            if return_annotated_image and detected_objects:
                np.random.seed(42)
                colors = np.random.randint(0, 255, size=(20, 3), dtype=np.uint8)
                if has_masks:
                    for out_idx, orig_idx in enumerate(final_kept_indices):
                        binary_mask = results[0].masks.data[orig_idx].cpu().numpy().astype(np.uint8)
                        if binary_mask.shape[0] != img_h or binary_mask.shape[1] != img_w:
                            binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        colored_mask = np.zeros_like(image, dtype=np.uint8)
                        colored_mask[binary_mask == 1] = colors[out_idx % len(colors)].tolist()
                        cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0, dst=image)

                for out_idx, obj in enumerate(detected_objects):
                    x1, y1, x2, y2 = obj["box"] if "box" in obj else [0,0,0,0]
                    color = colors[out_idx % len(colors)].tolist()
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(image, f"{obj['label']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                success, encoded_image = cv2.imencode('.jpg', image)
                if success:
                    annotated_image_b64 = base64.b64encode(encoded_image).decode('utf-8')

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
                "message": str(frame_err)
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"status": "success", "batch_results": compiled_batch_results}

# Register with RunPod Serverless
runpod.serverless.start({"handler": handler})




