# NVIDIA Device Plugin

The device plugin is a small daemon that runs on each GPU node and acts as a bridge between the GPU hardware and Kubernetes.

Kubernetes was designed for CPU and memory — it has no built-in concept of GPUs, FPGAs, or any custom hardware. The device plugin framework is how Kubernetes learns about non-standard hardware.

## The problem it solves

Without the device plugin:

- Pod requests: `nvidia.com/gpu: 1`
- Kubernetes: "I have no idea what `nvidia.com/gpu` is"
- Result: Pod stays `Pending` forever

With the device plugin:

- Device plugin tells kubelet: "This node has `nvidia.com/gpu: 1`"
- Pod requests: `nvidia.com/gpu: 1`
- Kubernetes: "Node X has it available, schedule there"
- Result: Pod runs on GPU node

## What it actually does

The device plugin performs three main tasks:

1. **Discovery**
   - Scans the node for NVIDIA GPUs using NVML (NVIDIA Management Library)
   - Reports available resources to kubelet
   - Example: `This node has nvidia.com/gpu: 1`

2. **Allocation**
   - When a pod requests a GPU, kubelet calls the device plugin
   - The plugin responds with the device assignment
   - Example: `Set NVIDIA_VISIBLE_DEVICES=0` in the container environment
   - That environment variable tells the container which physical GPU it can use

3. **Health monitoring**
   - Continuously checks GPU health
   - Marks GPUs unavailable if they fail
   - Prevents Kubernetes from scheduling new pods onto broken hardware

## The full picture

```
NVIDIA GPU hardware
        ↓
NVIDIA drivers (installed by UseGPUDedicatedVHD=true)
        ↓
NVML library (reads hardware counters, checks health)
        ↓
nvidia-device-plugin (translates NVML → Kubernetes API)
        ↓
kubelet (schedules pods based on available resources)
        ↓
Your pod gets nvidia.com/gpu: 1
```

The drivers talk to hardware. The device plugin talks to Kubernetes. They are two different layers solving two different problems, which is why AKS installs one but not the other.
