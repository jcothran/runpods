import os
import time
import base64
import json
import requests
import shutil
import tarfile

# --- CONFIGURATION ---
RUNPOD_API_KEY = "xxx" #use your runpod API key here
ENDPOINT_ID = "xxx" #use your endpoint ID here

OUTPUT_IMAGE_FOLDER = "annotated_results"
LOCAL_EXEMPLAR_PATH = "water_sample.png"

TELEMETRY_JSONL_FOLDER_IN = "./jsonl"
TELEMETRY_JSONL_FOLDER_OUT = "../jsonl"
ARCHIVE_FOLDER = "./jsonl/archive"

# Local path where the webserver exposes assets publicly
LOCAL_TAR_OUTPUT_PATH = "/var/www/html/webcoos.tar"

BATCH_LIMIT = 100            # 🚀 Back to high performance batch counts

ALLOWED_CAMERA_IDS = [
    "currituck_hampton_inn", "currituck_sailfish", "northinlet", "tmmc_prls",
    "carovabeach", "oceancity", "vabeach_hamptonos", "cms_dock_north",
    "cms_dock_south", "folly6thavenue", "masonboro_inlet"
]

RUN_URL = f"https://api.runpod.ai/v1/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v1/{ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}
# ---------------------

def derive_image_url_from_jsonl(jsonl_file_path):
    try:
        with open(jsonl_file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            data = json.loads(content)
        return data['data']['properties']['url']
    except Exception as e:
        print(f"❌ Failed parsing source URL from {jsonl_file_path}: {e}")
        return None

def get_active_prompts_for_url(image_url):
    if 'northinlet' in image_url:
        return ["bird"]
    if 'tmmc_prls' in image_url:
        return ["seal"]
    if 'cms_dock' in image_url:
        return ["boat"]
    return ["person", "umbrella", "chair"]

def download_image_locally(url, dest_folder):
    try:
        os.makedirs(dest_folder, exist_ok=True)
        filename = url.split('/')[-1].split('?')[0]
        local_path = os.path.join(dest_folder, filename)

        if not os.path.exists(local_path):
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
        return local_path if os.path.exists(local_path) else None
    except Exception as e:
        print(f"❌ Failed local cache acquisition for {url}: {e}")
        return None

def process_batch_jobs(batch_tasks):
    # Optional Exemplar Encoding
    exemplar_b64_string = None
    if LOCAL_EXEMPLAR_PATH and os.path.exists(LOCAL_EXEMPLAR_PATH):
        with open(LOCAL_EXEMPLAR_PATH, "rb") as img_f:
            exemplar_b64_string = base64.b64encode(img_f.read()).decode('utf-8')

    batch_inputs = []
    tarred_count = 0
    local_img_dir = "./img"

    # Write the tar ball directly to your web server directory block

    # --- FIXED ATOMIC WRITE PATTERN ---
    try:
        os.makedirs(os.path.dirname(LOCAL_TAR_OUTPUT_PATH), exist_ok=True)
        # 🛠️ Stage file under a temporary extension name first
        staging_tar_path = LOCAL_TAR_OUTPUT_PATH + ".tmp"

        with tarfile.open(staging_tar_path, mode="w") as tar:
            for file_path, filename, image_url in batch_tasks:
                derived_image_name = image_url.split('/')[-1].split('?')[0]
                local_path = os.path.join(local_img_dir, derived_image_name)

                if not os.path.exists(local_path):
                    local_path = download_image_locally(image_url, local_img_dir)

                if local_path and os.path.exists(local_path):
                    tar.add(local_path, arcname=derived_image_name)
                    tarred_count += 1

                batch_inputs.append({
                    "image_path": image_url,
                    "local_archive_filename": derived_image_name,
                    "prompts": get_active_prompts_for_url(image_url),
                    "source_filename": filename
                })

        # 🔐 Atomic swap: Instantly promote the file only after it is 100% closed
        if os.path.exists(staging_tar_path):
            os.replace(staging_tar_path, LOCAL_TAR_OUTPUT_PATH)
            print(f"🚀 Successfully saved complete archive out to webroot: {LOCAL_TAR_OUTPUT_PATH} ({tarred_count} images)")
        else:
            raise FileNotFoundError("Staging tar archive file failed write generation steps.")

    except Exception as tar_err:
        print(f"❌ Critical file engine permissions issue writing to web root: {tar_err}")
        return False


    '''
    try:
        os.makedirs(os.path.dirname(LOCAL_TAR_OUTPUT_PATH), exist_ok=True)
        with tarfile.open(LOCAL_TAR_OUTPUT_PATH, mode="w") as tar:
            for file_path, filename, image_url in batch_tasks:
                derived_image_name = image_url.split('/')[-1].split('?')[0]
                local_path = os.path.join(local_img_dir, derived_image_name)

                if not os.path.exists(local_path):
                    local_path = download_image_locally(image_url, local_img_dir)

                if local_path and os.path.exists(local_path):
                    tar.add(local_path, arcname=derived_image_name)
                    tarred_count += 1

                batch_inputs.append({
                    "image_path": image_url,
                    "local_archive_filename": derived_image_name,
                    "prompts": get_active_prompts_for_url(image_url),
                    "source_filename": filename
                })
        print(f"🚀 Successfully saved archive out to webroot: {LOCAL_TAR_OUTPUT_PATH} ({tarred_count} images)")
    except Exception as tar_err:
        print(f"❌ Critical file engine permissions issue writing to web root: {tar_err}")
        return False
    '''

    payload = {
        "input": {
            "batch_items": batch_inputs,
            "use_tar_url": tarred_count > 0,   # Tells handler whether to grab the host tar bundle
            "exemplar_image_b64": exemplar_b64_string,
            "return_annotated_image": False,
            "return_boxes": True,
            "return_polygons": True
        }
    }

    try:
        response = requests.post(RUN_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach RunPod: {e}")
        return False

    job_id = response.json().get("id")
    print(f"🆔 Sent Batch Job! ID: {job_id} | Waiting for GPU processing...")

    while True:
        try:
            status_response = requests.get(f"{STATUS_URL}/{job_id}", headers=HEADERS)
            status_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Polling hiccup: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        status_data = status_response.json()
        current_status = status_data.get("status")

        if current_status == "COMPLETED":
            results = status_data.get("output", {})
            if results.get("status") == "error":
                print(f"❌ Container Error: {results.get('message')}")
                return False

            batch_outputs = results.get("batch_results", [])
            print(f"⚡ Success! GPU batch completed. Processing {len(batch_outputs)} frame responses locally...")

            for item in batch_outputs:
                img_url = item.get("image_path")
                src_filename = item.get("source_filename")
                predictions = item.get("predictions", [])

                url_parts = img_url.split('/')
                try:
                    groups_index = url_parts.index("groups")
                    asset_name = url_parts[groups_index + 1]
                except (ValueError, IndexError):
                    asset_name = src_filename.split('_')[0]

                derived_image_name = url_parts[-1].split('?')[0]

                legacy_bboxes = []
                legacy_scores = []
                active_prompts = get_active_prompts_for_url(img_url)

                for obj in predictions:
                    box = obj.get("box")
                    if box:
                        legacy_bboxes.append([
                            {"x": box[0], "y": box[1]},
                            {"x": box[2], "y": box[3]}
                        ])
                    label = obj.get("label")
                    if label == "Visual Exemplar":
                        label = active_prompts[0]
                    legacy_scores.append({label: obj.get("conf", 1.0)})

                output_telemetry_payload = {
                    "data": {
                        "content": [
                            {
                                "classification_result": {
                                    "detected": len(predictions) > 0,
                                    "detection_count": len(predictions),
                                    "classification_bboxes": legacy_bboxes,
                                    "classification_scores": legacy_scores,
                                    "classification_model_name": "sam3",
                                    "classification_model_version": "native"
                                },
                                "original_image_reference": derived_image_name
                            }
                        ]
                    }
                }

                output_jsonl_name = f"{asset_name}_sam3_{derived_image_name}.jsonl"
                output_jsonl_path = os.path.join(TELEMETRY_JSONL_FOLDER_OUT, output_jsonl_name)

                with open(output_jsonl_path, "w") as jsonl_file:
                    json.dump(output_telemetry_payload, jsonl_file)
                print(f"💾 Telemetry log written: {output_jsonl_name}")

                b64_image_data = item.get("annotated_image_b64")
                if b64_image_data:
                    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
                    output_jpg_name = f"sam3_{derived_image_name}"
                    output_jpg_path = os.path.join(OUTPUT_IMAGE_FOLDER, output_jpg_name)
                    with open(output_jpg_path, "wb") as img_file:
                        img_file.write(base64.b64decode(b64_image_data))

            os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
            for file_path, filename, _ in batch_tasks:
                try:
                    shutil.move(file_path, os.path.join(ARCHIVE_FOLDER, filename))
                    print(f"📦 Input file archived cleanly: {filename}")
                except Exception as e:
                    print(f"⚠️ Error archiving {filename}: {e}")
            return True

        elif current_status == "FAILED":
            print(f"❌ RunPod Worker Failed: {status_data.get('error')}\n")
            return False

        elif current_status in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(1)

def scan_jsonl_folder_and_process():
    if not os.path.exists(TELEMETRY_JSONL_FOLDER_IN):
        return

    all_files = os.listdir(TELEMETRY_JSONL_FOLDER_IN)
    qualified_tasks = []

    for filename in all_files:
        if filename.endswith('.jsonl') and 'sam3' not in filename:
            if any(camera_id in filename for camera_id in ALLOWED_CAMERA_IDS):
                file_path = os.path.join(TELEMETRY_JSONL_FOLDER_IN, filename)
                url = derive_image_url_from_jsonl(file_path)
                if url:
                    qualified_tasks.append((file_path, filename, url))
                else:
                    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
                    shutil.move(file_path, os.path.join(ARCHIVE_FOLDER, filename))

    if not qualified_tasks:
        print("📭 No new qualifying input files found.")
        return

    for i in range(0, len(qualified_tasks), BATCH_LIMIT):
        current_batch = qualified_tasks[i:i + BATCH_LIMIT]
        print(f"\n📦 Processing batch package {(i//BATCH_LIMIT)+1} ({len(current_batch)} items)...")
        process_batch_jobs(current_batch)

if __name__ == "__main__":
    scan_jsonl_folder_and_process()
