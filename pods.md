# RunPod GPU Selection Guide: Moving Beyond Serverless Execution Limits

If your workloads are timing out or failing near the 5-minute mark, you are currently utilizing **RunPod Serverless**. RunPod Serverless is explicitly built for rapid, event-driven request cycles (like API-based model inference) and enforces automatic execution timeouts to prevent runaway background tasks.

For execution requirements extending from **an hour to several days** (e.g., model fine-tuning, large-scale batch processing, training pipelines), you must transition from Serverless to **RunPod Pods (Cloud GPUs)**.

---

## 1. RunPod Serverless vs. RunPod Pods

| Capability / Feature | RunPod Serverless | RunPod Pods (Cloud GPUs) |
| :--- | :--- | :--- |
| **Primary Use Case** | API endpoints, micro-inferences, quick automated jobs | Persistent computation, deep learning training, batch pipelines |
| **Execution Limits** | Strict timeouts (typically ~5–10 mins max per request) | **Unlimited** (runs continuously until manually terminated) |
| **Billing Model** | Millisecond-level granularity (only active execution) | Hourly fixed rate (active flat fee while container runs) |
| **Storage Persistence** | Ephemeral (wiped completely post-execution) | Persistent Volumes (retains data when stopped) |
| **Access Methods** | Webhooks, REST API calls, SDK triggers | Interactive Jupyter Lab, SSH terminal, custom HTTP ports |

---

## 2. Choosing Your Dedicated GPU Tier (Pods)

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

## 3. Workflow Adjustments for Long-Running Tasks

To pivot your setup seamlessly from Serverless to Pods, implement the following operational adjustments:

1. **Remove Handler Constraints:** You no longer need to package your entry script with a specialized serverless wrapper (e.g., `runpod.serverless.start()`). You can run your raw Python scripts (`python train.py`) directly inside the workspace.
2. **Utilize Persistent Volumes:** Allocate a persistent storage block when launching your Pod. If you pause the GPU to take a break or review logs, your files, model checkpoints, and weights remain securely stored. You will only pay a nominal storage fee (fractional cents per GB/month) while the GPU itself is stopped.
3. **Automate Shutdown Sequences:** Because Pods charge a flat hourly rate while active, a running instance can deplete your credits if left unattended. If your script takes exactly 1.5 hours to run, consider appending a programmatic teardown or webhook at the end of your python script, or monitor it via RunPod's CLI tool.

---

## 4. Quick-Start Checklist to Launch a Pod

1. Log into your **RunPod Console** and navigate to **Manage -> Pods**.
2. Click **Deploy** and toggle between *Secure Cloud* or *Community Cloud* depending on your budget and data security compliance.
3. Select your target GPU configuration and quantity.
4. Pick a deployment template (e.g., `RunPod PyTorch`, `RunPod TensorFlow`, or your own custom Docker container image).
5. Configure your **Volume Disk** and **Container Disk** allocations.
6. Click **Deploy**. Within 1–2 minutes, your persistent workspace will be live with Jupyter and SSH access ready.
