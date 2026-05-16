# Quantization Ladder — Mistral 7B on a T4 GPU

**Objective:** Running the same model at four precision levels on one GPU. Measuring memory and quality at each step.

---

## Overview

This project takes one model — Mistral 7B — and runs it at four precision levels on the same T4 GPU. At each level we measure memory used and quality produced. The goal is to build a complete comparison table from real measurements on real hardware.

The project answers three questions faced constantly in production AI infrastructure:

- How much memory does quantization actually save?
- What quality do you lose at each precision level?
- Which quantization tool is best for which use case?

| Phase | Precision | Tool | What I learn |
|---|---|---|---|
| 1 — Baseline OOM | FP16 | Native PyTorch | Why quantization exists — the problem statement |
| 2 — INT8 | 8-bit | bitsandbytes | First working model — 2x memory reduction |
| 3 — INT4 NF4 | 4-bit | bitsandbytes | 3.5x reduction, small quality tradeoff |
| 4 — INT4 AWQ | 4-bit | AutoAWQ | Best quality at INT4 — production standard |
| 5 — vLLM Serving | 4-bit AWQ | vLLM | Serving the quantized model in production pattern |
| 6 — Comparison | All | All | Full table with your real numbers |


---

## Background — Why Mistral 7B

### Model selection rationale

| | Mistral 7B | Llama 3 8B | Phi-3 14B |
|---|---|---|---|
| FP16 size | ~14 GB | ~16 GB | ~28 GB |
| Fits T4 in FP16? | ❌ Just over | ❌ Just over | ❌ Way over |
| Fits T4 in INT8? | ✅ ~7 GB | ✅ ~8 GB | ❌ ~14 GB |
| Fits T4 in INT4? | ✅ ~4 GB | ✅ ~5 GB | ✅ ~7 GB |
| INT4 quality | Excellent | Good | Good |
| AWQ support | Best in class | Good | Limited |

> **Key insight:** Mistral 7B uses Grouped Query Attention — an architectural feature that makes it remarkably robust to quantization. Quality degradation at INT4 is minimal compared to competitors. This is why it is the standard benchmark model for quantization research.

### The three quantization tools

| Tool | Method | Setup needed | Quality | Best for |
|---|---|---|---|---|
| bitsandbytes | On-the-fly quantization. One flag in HuggingFace. | None — just pip install | Good | Quick experiments, prototyping |
| GPTQ | Quantizes using calibration dataset to minimise accuracy loss. | Calibration dataset required | Better than bnb | When you control the quantization process |
| AWQ | Activation-aware — protects weights that matter most based on activation magnitudes. | Pre-quantized models on HuggingFace | Best at INT4 | Production serving — current state of the art |

For this project we use bitsandbytes for INT8 and INT4 NF4, and AWQ for the best-quality INT4. This gives you hands-on experience with all three approaches and a direct quality comparison between INT4 methods.

### The quality measurement

To measure quality impact we use a consistent set of factual questions with known correct answers. The model either gets it right or wrong, giving a clear accuracy percentage to compare across precision levels. We use 5 questions per run for speed on the T4, covering general knowledge, arithmetic, and factual recall — the categories most sensitive to quantization errors.

---

## Prerequisites

### Create the AKS cluster

```bash
az login

az group create --name gpu-aks-rg --location centralindia

az aks create \
  --resource-group gpu-aks-rg \
  --name gpu-aks \
  --node-count 1 \
  --node-vm-size Standard_D2s_v3 \
  --nodepool-name systempool \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --network-plugin azure
```

### Add the GPU node pool

```bash
az aks nodepool add \
  --resource-group gpu-aks-rg \
  --cluster-name gpu-aks \
  --name gpupool \
  --node-count 1 \
  --node-vm-size Standard_NC4as_T4_v3 \
  --node-taints sku=gpu:NoSchedule \
  --aks-custom-headers UseGPUDedicatedVHD=true
```

`--node-taints sku=gpu:NoSchedule` prevents regular pods from landing on the expensive GPU node.

`UseGPUDedicatedVHD=true` tells AKS to use a GPU-optimised node image with NVIDIA drivers pre-installed — no manual driver installation needed.

### Connect kubectl

```bash
az aks get-credentials --resource-group gpu-aks-rg --name gpu-aks
kubectl get nodes -o wide
```

### Install the NVIDIA device plugin

Always run this after the nodepool add. The device plugin is the component that tells Kubernetes about the GPU resource (`nvidia.com/gpu`) — without it pods stay Pending forever even though drivers are installed.

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --set 'tolerations[0].key=sku' \
  --set 'tolerations[0].operator=Equal' \
  --set 'tolerations[0].value=gpu' \
  --set 'tolerations[0].effect=NoSchedule' \
  --set 'nodeSelector.accelerator=nvidia' \
  --set nfd.enabled=false \
  --set gfd.enabled=false
```

If the device plugin is already installed, use `helm upgrade` instead of `helm install`.

Verify the GPU is visible to Kubernetes:

```bash
kubectl describe node <gpu-node-name> | grep "nvidia.com/gpu"
# Expected: nvidia.com/gpu: 1
```

> **After node pool scale-up:** If `nvidia.com/gpu` is missing after scaling, apply these labels manually:
> ```bash
> kubectl label node <node-name> \
>   "feature.node.kubernetes.io/pci-10de.present=true" \
>   "nvidia.com/gpu.present=true"
> ```

### HuggingFace token

Create a HuggingFace account to download Mistral 7B:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the Mistral 7B model license at [huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---

## Observability Setup

### GPU verification pod

```bash
kubectl apply -f manifests/nvidia-smi-pod.yaml
kubectl logs nvidia-smi-pod
```

### dcgmi observer

Apply `manifests/dcgmi-observer.yaml` — streams live GPU metrics every 500ms showing temperature, power, SM utilization, memory utilization, framebuffer used.

```bash
kubectl apply -f manifests/dcgmi-observer.yaml
kubectl logs dcgmi-observer -f
```

### Prometheus + Grafana stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.scrapeInterval="30s" \
  --set grafana.adminPassword="admin123"

kubectl get pods -n monitoring --watch
```

### dcgm-exporter

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --set "tolerations[0].key=sku" \
  --set "tolerations[0].operator=Equal" \
  --set "tolerations[0].value=gpu" \
  --set "tolerations[0].effect=NoSchedule" \
  --set "nodeSelector.accelerator=nvidia" \
  --set "serviceMonitor.enabled=true"
```

Fix the ServiceMonitor label so Prometheus scrapes it:

```bash
kubectl label servicemonitor dcgm-exporter -n monitoring release=prometheus
```

### Grafana dashboard

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &
```

Open `http://localhost:3000` — login: `admin` / `admin123`

Import the NVIDIA DCGM dashboard via API:

```bash
curl -s https://grafana.com/api/dashboards/12239/revisions/latest/download -o nvidia-dashboard.json

python3 -c "
import json
with open('nvidia-dashboard.json') as f:
    d = json.load(f)
content = json.dumps(d).replace('\${DS_PROMETHEUS}', 'prometheus')
with open('nvidia-dashboard-fixed.json', 'w') as f:
    f.write(content)
"

curl -s -X POST \
  http://admin:admin123@localhost:3000/api/dashboards/import \
  -H 'Content-Type: application/json' \
  -d "{\"dashboard\": $(cat nvidia-dashboard-fixed.json), \"overwrite\": true, \"folderId\": 0, \"inputs\": [{\"name\": \"DS_PROMETHEUS\", \"type\": \"datasource\", \"pluginId\": \"prometheus\", \"value\": \"prometheus\"}]}"
```

| Panel | Metric | What it shows |
|---|---|---|
| GPU Utilization | `DCGM_FI_DEV_GPU_UTIL` | % of time GPU was computing |
| Tensor Core Utilization | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core pipeline activity |
| SM Clock | `DCGM_FI_DEV_SM_CLOCK` | Compute clock — jumps during workload |
| VRAM Used | `DCGM_FI_DEV_FB_USED` | Memory consumed by workloads |
| Power Usage | `DCGM_FI_DEV_POWER_USAGE` | Watts drawn — correlates with compute |
| Temperature | `DCGM_FI_DEV_GPU_TEMP` | GPU core temperature |

> **Stale series fix:** If panels show inflated values from completed pods, edit the query to `last_over_time(DCGM_FI_DEV_POWER_USAGE{gpu="0"}[2m])`.

---

## Phase 1 — Baseline OOM (FP16)

This phase is intentional failure. We try to load Mistral 7B in FP16, watch it hit the VRAM limit, and capture the exact memory reading. This is the problem statement for the entire project.

> **Warning:** This pod is designed to crash or run at the absolute VRAM limit. That is the expected and correct outcome. The GPU itself is not harmed.

### Deploy

Create the script ConfigMap:

```bash
kubectl create configmap phase1-script --from-file=phase1.py=scripts/phase1.py
```

Apply the pod:

```bash
kubectl apply -f manifests/mistral-fp16-oom.yaml
kubectl logs mistral-fp16-oom -f
```

### Expected output

```
GPU: Tesla T4
Total VRAM: 15.8 GB
Attempting to load Mistral 7B in FP16...
FAILED: CUDA out of memory. Tried to allocate 1.17 GiB.
Memory at crash: 14.94 GB / 15.8 GB
This is exactly why quantization exists.
```

### What to watch in dcgmi

```
#Entity   Temp  Power  SMUtil  FBUsed
GPU 0      36    18      8      2048    <- weights loading
GPU 0      38    22     12      8192    <- halfway
GPU 0      41    28     15     14900    <- near limit
GPU 0      41    12      0     14900    <- crash, pod dead
```

### Measurements

| Metric | Expected | My measurement |
|---|---|---|
| FP16 memory at load | ~14.9 GB | 15.02 GB |
| FP16 memory after inference | ~15.0 GB | 15.03 GB |
| T4 total VRAM (PyTorch reported) | 15.8 GB | 16.7 GB |
| Headroom after inference | ~0.9 GB | 0.77 GB |
| Pod crashed? | Yes (expected OOM) | No — but unusable in production |

![FP16 pod logs and results](screenshots/fp16-results.png)

![Grafana GPU metrics during FP16 load](screenshots/fp16-grafana.png)

> **Note on the 16.7 GB figure:** PyTorch's `get_device_properties().total_memory` includes reserved system memory. The usable VRAM is effectively 15.8 GB matching the T4 spec.

**Key takeaway:** 0.77 GB headroom is not enough for production — no batching, no long contexts, one slightly longer prompt away from OOM. INT8 should cut this to ~7.5 GB used, giving 8+ GB of headroom.

---

## Phase 2 — INT8 with bitsandbytes

One flag changes everything. `load_in_8bit=True` quantizes weights on the fly as they load from disk to GPU. Every weight that was 16 bits becomes 8 bits. The model footprint halves.

> **Key insight:** bitsandbytes uses LLM.int8() under the hood, which handles outlier activations specially. Outlier weights (unusually large values) stay in FP16 while the rest go to INT8. This is what prevents the quality collapse you would expect from naive 8-bit rounding.

### Deploy

```bash
kubectl create configmap phase2-script --from-file=phase2.py=scripts/phase2.py
kubectl apply -f manifests/mistral-int8.yaml
kubectl logs mistral-int8 -f
```

### Expected output

```
Loading Mistral 7B in INT8...
Memory used: 7.21 GB
Memory saved vs FP16 baseline (~14GB): 6.8 GB saved
Q: What is the capital of France?
A: Paris | Expected: Paris | Correct: True | 312ms
INT8 Accuracy: 5/5 = 100%
Avg latency: 308ms
Peak memory: 7.45 GB
```

### Measurements

| Metric | Expected | My measurement |
|---|---|---|
| INT8 memory used | ~7.2 GB | 8.05 GB |
| Memory reduction vs FP16 | ~2x | 6 GB saved |
| Accuracy (5 questions) | 5/5 | 5/5 |
| Avg latency | ~300ms | 2396ms |

![INT8 pod logs and results](screenshots/int8-results.png)

![Grafana GPU metrics during INT8 inference](screenshots/int8-grafana.png)

> **Note on latency:** bitsandbytes LLM.int8() uses mixed-precision decomposition that doesn't map cleanly to T4's INT8 Tensor Cores. Some computation stays in FP16, resulting in higher latency than expected. INT8 solves the memory problem but creates a latency problem on T4 — this sets up the case for INT4.

---

## Phase 3 — INT4 NF4 with bitsandbytes

Drop to 4-bit. NF4 (NormalFloat4) is the best 4-bit format for normally-distributed weights — which is exactly what transformer weights are. `double_quant=True` applies a second quantization to the quantization constants themselves, saving another ~0.4 GB.

> **Key insight:** NF4 is not the same as INT4. Regular INT4 uses uniformly spaced values. NF4 uses values spaced to match the normal distribution of typical neural network weights — more precision where values cluster (near zero), less where values are rare (far from zero). This is why NF4 quality is much better than naive INT4.

### Deploy

```bash
kubectl create configmap phase3-script --from-file=phase3.py=scripts/phase3.py
kubectl apply -f manifests/mistral-int4-nf4.yaml
kubectl logs mistral-int4-nf4 -f
```

### Expected output

```
Loading Mistral 7B in INT4 NF4...
Memory used: 4.18 GB
Q: What is the capital of France? | Correct: True | 198ms
INT4 NF4 Accuracy: 5/5 = 100%
Avg latency: 197ms
Peak memory: 4.52 GB
```

### Measurements

| Metric | Expected | My measurement |
|---|---|---|
| INT4 NF4 memory used | ~4.2 GB | 4.66 GB |
| Memory reduction vs FP16 | ~3.5x | 15.03 – 4.66 = 10.37 GB saved |
| Memory reduction vs INT8 | ~1.7x | 8.05 – 4.66 = 3.39 GB saved |
| Accuracy (5 questions) | 4–5/5 | 4/5 |
| Avg latency | ~200ms | 1097ms |

![INT4 NF4 pod logs and results](screenshots/int4-results.png)

---

## Phase 4 — INT4 AWQ (Best Quality)

AWQ (Activation-aware Weight Quantization) is the current state of the art for INT4. Instead of treating all weights equally, AWQ analyses which weights are most important based on activation magnitudes and protects those during quantization.

Pre-quantized AWQ models already exist on HuggingFace. We use `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` — no calibration needed.

> **Key insight:** AWQ typically gives 1–3% better accuracy than bitsandbytes NF4 at the same 4-bit precision. For simple factual questions you may not see a difference, but on harder reasoning tasks the gap is measurable. This is why AWQ is the standard for production INT4 serving.

### Deploy

```bash
kubectl create configmap phase4-script --from-file=phase4.py=scripts/phase4.py
kubectl apply -f manifests/mistral-awq.yaml
kubectl logs mistral-awq -f
```

> **Note on dependencies:** The `manifests/mistral-awq.yaml` pod uses `autoawq-kernels` for optimised CUDA inference. If the kernel install fails on your environment, fall back to the bitsandbytes INT4 NF4 approach from Phase 3 — same memory footprint, marginally lower quality.

### Measurements

| Metric | Expected | My measurement |
|---|---|---|
| AWQ INT4 memory used | ~4.0 GB | 4.66 GB |
| Accuracy (5 questions) | 5/5 | 4/5 |
| Avg latency | ~160ms | 1085ms |
| Quality vs NF4 | Same or better | Same |

---

## Phase 5 — Serve AWQ with vLLM

Now that we have the best-quality INT4 model, serve it properly with vLLM — the same production inference engine, now serving a real 7B model at 4-bit precision.

> **Key insight:** vLLM has native AWQ support. The `--quantization awq` flag tells vLLM to use its optimised AWQ CUDA kernels rather than the generic HuggingFace path. This gives significantly better throughput than the direct HuggingFace approach in Phase 4.

### Deploy

```bash
kubectl apply -f manifests/mistral-vllm-awq.yaml
kubectl logs -f mistral-vllm-awq
# Wait for: INFO: Application startup complete.
```

![vLLM model loading output](screenshots/mistal-vllm-awq-loading.png)

### Send inference requests

```bash
kubectl port-forward pod/mistral-vllm-awq 8000:8000 &

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    "messages": [{"role": "user", "content": "Explain GPU quantization in 3 sentences."}],
    "max_tokens": 150
  }'
```

![vLLM single inference response](screenshots/mistal-vllm-awq-inference.png)

Concurrent requests test:

```bash
for i in {1..5}; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
         "messages": [{"role": "user", "content": "What is machine learning?"}],
         "max_tokens": 100}' &
done
wait
echo "All requests complete"
```

### What to watch in dcgmi during vLLM serving

```
#Entity   Temp  Power  SMUtil  FBUsed   TCUtil
GPU 0      36     9      0       0         0    <- before load
GPU 0      44    38     42    4200        35    <- model loaded, serving
GPU 0      48    44     55    4200        52    <- concurrent requests
```

Key observation: FBUsed here (~4,200 MiB) vs the FP16 crash (~14,900 MiB). Same model. 3.5x less memory. This is quantization working in production.

![dcgmi live metrics during vLLM serving](screenshots/mistal-vllm-awq-dcgmi-results.png)

![Grafana dashboard during vLLM serving](screenshots/mistal-vllm-awq-grafana.png)

---

## Phase 6 — The Comparison Table

### Memory comparison

| Precision | Tool | Memory used | vs FP16 | Fits T4? |
|---|---|---|---|---|
| FP16 | PyTorch | ~14.9 GB (OOM) | 1x baseline | ❌ Fits however can OOM with few inferences |
| INT8 | bitsandbytes | 8.05 GB | 0.54x | ✅ |
| INT4 NF4 | bitsandbytes | 4.66 GB | 0.31x | ✅ |
| INT4 AWQ | AutoAWQ | 4.66 GB | 0.31x | ✅ |
| INT4 AWQ | vLLM | ~4.2 GB | 0.28x | ✅ |

### Quality and latency comparison

| Precision | Tool | Accuracy (5 Qs) | Avg latency | Notes |
|---|---|---|---|---|
| FP16 | PyTorch | N/A (OOM) | N/A | Crashed in few inference |
| INT8 | bitsandbytes | 5/5 | 2396ms | LLM.int8() protects outliers |
| INT4 NF4 | bitsandbytes | 4/5 | 1097ms | NormalFloat4 + double quant |
| INT4 AWQ | AutoAWQ | 4/5 | 1085ms | Activation-aware, best quality |
| INT4 AWQ | vLLM | N/A | N/A | Throughput-optimised serving |

### Key conclusions

After filling in the table, few questions are answered:

- How much memory does halving precision actually save? (INT8 vs FP16 vs INT4)
  - FP16 almost consumes 16 GB of memory and the INT8 and INT4 consumes 50% less memeory respectively.
- Is there a measurable accuracy difference between INT8 and INT4 NF4 on simple questions?
  - Both were able to answer 4/5 sampling questions correctly.
- Is AWQ measurably better than NF4 at the same bit-width?
  - Yes, it does.

---

## Cleanup

```bash
# Delete all pods
kubectl delete pod mistral-fp16-oom mistral-int8 mistral-int4-nf4 mistral-awq mistral-vllm-awq

# Delete ConfigMaps
kubectl delete configmap phase1-script phase2-script phase3-script phase4-script

# Scale GPU node to zero (stops VM billing)
az aks nodepool scale \
  --resource-group gpu-aks-rg \
  --cluster-name gpu-aks \
  --name gpupool \
  --node-count 0

# Full teardown
az group delete --name gpu-aks-rg --yes --no-wait
```