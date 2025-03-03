FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /app/static/uploads \
    && mkdir -p /opt/packages \
    && chown -R ${USERNAME}:${USERNAME} /app \
    && chown ${USERNAME}:${USERNAME} /opt/packages

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    python3-pip \
    python3-opencv \
    libglib2.0-0 \
    wget \
    build-essential \
    cmake \
    pkg-config \
    liblapack-dev \
    liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up pip for the user
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
RUN pip3 install --upgrade pip && \
    mkdir -p /home/${USERNAME}/.local/bin && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.local

# RTX 2080 Ti
ENV TORCH_CUDA_ARCH_LIST="7.5"

# Switch to non-root user for pip installations
USER $USERNAME

# Install PyTorch with CUDA support
RUN pip3 install --user --no-cache-dir torch torchvision torchaudio

# Clone and install SAM2 in /opt/packages
WORKDIR /opt/packages
RUN git clone https://github.com/facebookresearch/sam2.git && \
    cd sam2 && \
    pip3 install --user -e . && \
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ../..

# Switch back to app directory
WORKDIR /app

# Switch back to root for copying files (COPY needs root)
USER root

# Copy project files
COPY --chown=${USERNAME}:${USERNAME} . .
RUN chown -R ${USERNAME}:${USERNAME} /app/static/uploads

# Switch back to non-root user for final installation
USER $USERNAME

# Install dependencies
RUN pip3 install --user --no-cache-dir -e '.[dev]'

EXPOSE 8000
EXPOSE 5678

# CMD ["uvicorn", "src.click2mask.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]