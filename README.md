# Development Environment Setup Guide

## Prerequisites

Before you can start development, make sure you have the following installed:

1. Docker
2. NVIDIA GPU drivers
3. CUDA ([installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu))
4. NVIDIA Container Toolkit and configured Docker ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
5. Visual Studio Code with Remote Development extension

## NVIDIA Docker Runtime Setup

You can verify the setup by running:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

This should display your GPU information.