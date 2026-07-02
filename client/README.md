# Automated Coastal Monitoring Pipeline (WebCOOS & RunPod SAM3)

This repository contains an automated, distributed computer vision pipeline designed to pull coastal monitoring camera images from WebCOOS, package them for batch processing, and run object detection/segmentation tasks using [**Ultralytics SAM3 (Segment Anything Model 3)**](https://docs.ultralytics.com/models/sam-3) hosted on [**RunPod Serverless GPU infrastructure**](https://www.runpod.io/).

---

## 🏗️ System Architecture & Workflow

The system is split into a **Local Orchestrator** (managing scheduling, API polling, asset distribution, and file logging) and a **Remote Serverless GPU Worker** (handling deep learning inference).

```
[Local Host Crontab] ──► Triggering Bash Wrappers
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Phase 1: Metadata Scraping (json_latest_uuid.py)         │
  │ ├─ Polls WebCOOS camera endpoints every 10 minutes       │
  │ └─ Logs latest frame references to unique local .jsonl    │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Phase 2: Local Orchestration (client_pipeline_image.py) │
  │ ├─ Scans pending .jsonl files hourly                      │
  │ ├─ Downloads images & packs them into an atomic .tar      │
  │ └─ Dispatches payload to RunPod Serverless execution API  │
  └─────────────────────────────┬─────────────────────────────┘
                                │ (Secure HTTP Request)
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Phase 3: Remote GPU Compute ([handler.py](https://github.com/jcothran/runpods/blob/main/handler.py) / [Dockerfile](https://github.com/jcothran/runpods/blob/main/Dockerfile))     │
  │ ├─ Worker wakes up, downloads & extracts webcoos.tar      │
  │ ├─ Processes SAM3 model inference in VRAM via GPU         │
  │ └─ Returns coordinates & base64 annotated visualizations  │
  └─────────────────────────────┬─────────────────────────────┘
                                │ (API Response JSON)
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │ Phase 4: Finalization & Archiving                         │
  │ ├─ Writes final telemetry logs to host server webroot     │
  │ └─ Cleanly shifts processed metadata to input archive     │
  └───────────────────────────────────────────────────────────┘
```

---

## 🛠️ Components & File Manifest

### 1. Scheduler & Wrappers (Local Host Machine)

#### `crontab`
Manages execution timing inside the running Docker container container (`2b9`).
* **Metadata Polling:** Runs every 10 minutes (`0,10,20,30,40,50`) from 8:00 AM to 6:50 PM to query the newest camera asset references.
* **Pipeline Execution:** Runs once an hour at the top of the hour (`0`) from 9:00 AM to 7:00 PM to batch-process accumulated image frames.

#### `sh_json_latest_sam3` & `sh_json_track_sam3`
Lightweight bash utilities that ensure the environment context changes to `/var/www/html/sam3` before executing the respective Python sub-modules.

---

### 2. Python Source Files (Local Host Machine)

#### `json_latest_uuid.py`
* **Function:** Queries the WebCOOS API utilizing an authenticated token across a predetermined map of unique camera UUIDs (e.g., `uncw_masonboro_inlet`, `maracoos_oceancity`).
* **Output:** Creates the `jsonl/` directory if missing and dumps the latest unique timestamped capture information inside: `jsonl/{camera_filename}_{YYYYMMDD_HHMMSS}.jsonl`.

#### `client_pipeline_image.py`
The overarching orchestrator of the localized workflow loop.
* **Directory Scan:** Examines `./jsonl` for fresh file strings omitting the `'sam3'` processing signature.
* **Atomic Tar Assembly:** Downloads the original remote source image via standard HTTP GET, tracks them via an allowed camera whitelist, and packages them inside a compressed archive block at `/var/www/html/webcoos.tar`. Uses a `.tmp` staging process with an instant swap (`os.replace`) to guarantee zero race conditions on the web server.
* **Job Dispatch & Polling:** Transmits specialized prompt payloads (e.g., matching `"seal"` for wildlife hubs, or `"boat"` for maritime docks) to the RunPod cluster, then enters a defensive `while True` polling block monitoring the state of the GPU compute node.
* **Data Resolution:** Unpacks returned box models, structures unified telemetry output matrices, maps base64-encoded visual annotations back into image files, and pushes processed metadata blocks into `./jsonl/archive`.

---

### 3. Serverless Environment Infrastructure (RunPod Container)

#### `Dockerfile`
Defines the reproducible, isolated Linux image deployed out on the RunPod orchestration cluster.
* **Base Image:** Extends `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` to unlock standard system bindings for CUDA 12.8 and PyTorch 2.8.
* **Dependencies:** Pre-bakes essential deep learning support matrices (`ultralytics`, `timm`, `safetensors`) and compiles the `CLIP` natural language contextual visual framework natively via Git.
* **Initialization:** Commands the environment to trigger the service handler layer using unbuffered standard output routing (`python -u /app/handler.py`).

#### `handler.py`
The active Python runtime listening for jobs on the remote serverless GPU worker.
* **VRAM Warming Optimization:** Loads the foundational heavy segmentation model weights (`sam3.pt`) *globally* during container initialization into GPU device 0 VRAM. This removes model loading overhead from the operational execution loops, slashing execution times to under a second per inference action.
* **Optimized Network Intake:** Rather than bottlenecking the system by pulling 100 images sequentially via separate slow web streams, the handler downloads the unified `webcoos.tar` package directly from the master host (`saludasys.org`) and extracts it into `/tmp/extracted_images`.
* **Execution Paradigms:** Resolves objects via two distinct methods depending on your payload requirements:
    1.  **Text Prompting:** Searches for targets matching language structures like `"person"`, `"umbrella"`, or `"bird"`.
    2.  **Visual Exemplar Matching(optional):** Uses an encoded transparency-based target mask (`water_sample.png`) to segment structural features resembling the target sample matrix.
* **Post-processing Refinement:** Evaluates overlapping bounding box configurations using Non-Maximum Suppression (`cv2.dnn.NMSBoxes`) to discard low-confidence duplicate layers.
* **Result Structuring:** Formulates coordinate layouts, calculates precise pixel area sums, paints colored alpha masks over verified targets via OpenCV, converts the raw graphics array to an encapsulated base64 string, and passes the entire payload back down the WebSocket pipe to complete the job.

---

## 🚀 Key Optimization & Design Strategies

1.  **Serverless Cost-Efficiency:** VRAM warming handles heavy initialization overhead just once during cold boots. The compute worker only processes blocks when jobs populate the queue, saving significant idle GPU cost.
2.  **Network Bundle Aggregation:** Compressing sequential image arrays into a single cohesive tarball asset structure circumvents standard network request overheads, keeping data ingestion fast and predictable.
3.  **Atomic Persistence Safeties:** The pipeline never writes structures directly to production web directories. Staging assets under transient names (`webcoos.tar.tmp`) before calling explicit atomic system shifts avoids corrupting partial transfers if a step fails.
