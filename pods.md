# RunPod GPU Selection Guide: Moving Beyond Serverless Execution Limits

If your workloads are timing out or failing near the 5-minute mark, you are currently utilizing **RunPod Serverless**. RunPod Serverless is explicitly built for rapid, event-driven request cycles (like API-based model inference) and enforces automatic execution timeouts to prevent runaway background tasks.

For execution requirements extending from **an hour to several days** (e.g., model fine-tuning, large-scale batch processing, training pipelines), you must transition from Serverless to **RunPod Pods (Cloud GPUs)**.

---

## RunPod Serverless vs. RunPod Pods

| Capability / Feature | RunPod Serverless | RunPod Pods (Cloud GPUs) |
| :--- | :--- | :--- |
| **Primary Use Case** | API endpoints, micro-inferences, quick automated jobs | Persistent computation, deep learning training, batch pipelines |
| **Execution Limits** | Strict timeouts (typically ~5–10 mins max per request) | **Unlimited** (runs continuously until manually terminated) |
| **Billing Model** | Millisecond-level granularity (only active execution) | Hourly fixed rate (active flat fee while container runs) |
| **Storage Persistence** | Ephemeral (wiped completely post-execution) | Persistent Volumes (retains data when stopped) |
| **Access Methods** | Webhooks, REST API calls, SDK triggers | Interactive Jupyter Lab, SSH terminal, custom HTTP ports |

---

## Choosing Your Dedicated GPU Tier (Pods)

RunPod categorizes its infrastructure into **Secure Cloud** (tier-3 enterprise data centers) and **Community Cloud** (peer-to-peer hosting, highly cost-effective). Below are the recommended dedicated GPU configurations optimized for extended workloads:

### Consumer & Prosumer Tier (High Availability & Value)
* **NVIDIA RTX 4090 (24 GB VRAM)**
  * *Typical Cost:* ~$0.34 – $0.69 / hr
  * *Best For:* Light model fine-tuning (LoRAs), massive image generation batches, small-to-medium object detection pipelines.
* **NVIDIA RTX 6000 Ada (48 GB VRAM)**
  * *Typical Cost:* ~$0.74 – $0.99 / hr
  * *Best For:* Intermediate LLM customization, advanced multi-modal models, computer vision tasks requiring large batch size headroom.

### Enterprise & Data Center Tier (Maximum Throughput)
* **NVIDIA A100 / H100 (80 GB VRAM)**
  * *Typical Cost:* ~$1.19 – $2.50+ / hr
  * *Best For:* Massive deep learning architectures, heavy data engineering parallel operations, chunking and processing petabyte-scale datasets.

---

## Workflow Adjustments for Long-Running Tasks

To pivot your setup seamlessly from Serverless to Pods, implement the following operational adjustments:

1. **Remove Handler Constraints:** You no longer need to package your entry script with a specialized serverless wrapper (e.g., `runpod.serverless.start()`). You can run your raw Python scripts (`python train.py`) directly inside the workspace.
2. **Utilize Persistent Volumes:** Allocate a persistent storage block when launching your Pod. If you pause the GPU to take a break or review logs, your files, model checkpoints, and weights remain securely stored. You will only pay a nominal storage fee (fractional cents per GB/month) while the GPU itself is stopped.
3. **Automate Shutdown Sequences:** Because Pods charge a flat hourly rate while active, a running instance can deplete your credits if left unattended. If your script takes exactly 1.5 hours to run, consider appending a programmatic teardown or webhook at the end of your python script, or monitor it via RunPod's CLI tool.

---

## Quick-Start Checklist to Launch a Pod

1. Log into your **RunPod Console** and navigate to **Manage -> Pods**.
2. Click **Deploy** and toggle between *Secure Cloud* or *Community Cloud* depending on your budget and data security compliance.
3. Select your target GPU configuration and quantity.
4. Pick a deployment template (e.g., `RunPod PyTorch`, `RunPod TensorFlow`, or your own custom Docker container image).
5. Configure your **Volume Disk** and **Container Disk** allocations.
6. Click **Deploy**. Within 1–2 minutes, your persistent workspace will be live with Jupyter and SSH access ready.

---

##Automate RunPods to run on a daily schedule

To automate RunPod Pods to run on a daily schedule, because Pods don't have a native "cron schedule" button in the web UI, you have to use a little scripting.

The most robust way to do this is by using the RunPod Web API or Python SDK paired with a local task scheduler (like standard Cron on Linux/Mac, Task Scheduler on Windows, or a free GitHub Action).

Here are the two best ways to handle it, depending on whether you want to completely destroy the pod daily to save money, or keep your files intact between runs.

Method 1: The "Start & Stop" Approach (Saves Data)
If your container takes a long time to download weights or packages, you don't want to delete it every day. Instead, you create a Pod with a Persistent Volume. You only pay a few cents a day to store your data, and you programmatically turn the GPU on and off.

Step 1: Your Startup Script (Inside the Pod)
Set up your Python script inside the pod so that when it finishes its 1–2 hours of work, it automatically stops itself. You can do this by using the RunPod API to tell the system to halt the pod from the inside.

```python
import os
import requests

def main_work():
    # Your 1-2 hour data pipeline, training, or batch job goes here
    print("Running daily processing...")

if __name__ == "__main__":
    main_work()
    
    # --- AUTOMATIC SELF-SHUTDOWN ---
    # Retrieve the current Pod ID from the environment variables
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = "YOUR_RUNPOD_API_KEY"
    
    if pod_id:
        print(f"Job complete. Requesting shutdown for pod {pod_id}...")
        url = f"https://api.runpod.io/v1/pod/stop/{pod_id}?api_key={api_key}"
        requests.post(url)
```

Step 2: The Daily Trigger (Outside the Pod)
Now you just need an external trigger to wake the pod up every day. From your home machine, a local server, or a cloud function, schedule a simple API call to run daily at your preferred hour:

```bash
# Example Curl command to start your specific pod id
curl -X POST "https://api.runpod.io/v1/pod/start/YOUR_POD_ID?api_key=YOUR_RUNPOD_API_KEY"
```

Linux/Mac: Put this line into a standard crontab -e job scheduled for every morning.

Windows: Put it in a lightweight .bat script triggered by Task Scheduler.

Method 2: The "Create & Terminate" Approach (Saves the Most Money)
If your environment is entirely packaged into a custom Docker image and it doesn't need to save data between runs, you can save even more money by creating a fresh pod from scratch every day and terminating it entirely when done.

You can write a lightweight local Python script using the runpod library, and schedule it to run once a day on your local machine or via GitHub Actions:

```python
import runpod
import time

runpod.api_key = "YOUR_RUNPOD_API_KEY"

# 1. Spin up a fresh GPU pod
print("Launching daily GPU instance...")
pod = runpod.create_pod(
    name="Daily-Batch-Job",
    image_name="your-docker-username/your-image:latest",
    gpu_type_id="NVIDIA RTX 4090", # or RTX 5070 / 6000 Ada if available
    gpu_count=1,
    volume_in_gb=20
)

pod_id = pod['id']

# 2. Wait for your script to do its thing 
# (Alternatively, let the pod run its entrypoint script and just sleep here for 2 hours)
time.sleep(7200) 

# 3. Completely wipe the pod so billing stops entirely
print("Time limit reached. Terminating pod...")
runpod.terminate_pod(pod_id)
```

Which one should you pick?
Go with Method 1 (Start/Stop) if you are working with large ML models, heavy checkpoints, or environments that take longer than 5 minutes to initialize. It ensures you don't waste time downloading gigabytes of weights every single day.

Go with Method 2 (Create/Terminate) if your setup is ultra-lean, spins up instantly, and you want absolutely zero trailing storage fees on your account when the GPU is offline.
