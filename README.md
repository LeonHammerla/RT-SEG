<p align="center">
  <img src="assets/logo.svg" width="40%" style="max-width: 400px;">
</p>
This project requires a configuration file for database authentication.

### 🔐 Configuration Setup

Please ensure the `sdb_login.json` file is located in the `/data/` directory with the following structure:

```json
{
  "user": "", 
  "pwd": "", 
  "ns": "NR",
  "db": "RT",
  "url": "wss://domain"
}


# Running RT-SEG as a Dockerized GPU Application

This project can be run entirely inside Docker with NVIDIA GPU acceleration.

The container bundles CUDA **user-space libraries** and Python **3.12.3**.  
Your **host provides the NVIDIA driver**, which is injected into the container at runtime.

---

## Host Requirements

You must have:

- Linux (recommended)
- An NVIDIA GPU
- NVIDIA driver installed (`nvidia-smi` must work)
- Docker Engine
- NVIDIA Container Toolkit (for `--gpus all`)

Verify GPU access from Docker:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi


If the second command prints your GPU table, Docker GPU support is working.

CUDA Compatibility (Important)

The container is built on CUDA 12.x.

Your host driver must support at least CUDA 12.x.

This is a one-way compatibility rule:

Host driver CUDA version ≥ Container CUDA version

Examples:

Host Driver	Container CUDA	Result
12.8	12.4	✅ Works
12.8	12.8	✅ Works
12.8	13.1	❌ Fails
13.x	12.4	✅ Works

Check your host driver:

nvidia-smi


If you see:

nvidia-container-cli: requirement error: unsatisfied condition: cuda>=...


your container CUDA is newer than your driver. Either update your driver or rebuild using an older CUDA base.

This repository intentionally targets CUDA 12.x for broad compatibility.

Build the Docker Image

From the project root:

docker build -f docker/Dockerfile -t mytui:gpu .


This will:

Build Python 3.12.3 from source

Compile native dependencies (e.g. hdbscan)

Produce a slim runtime image

Run the Application

Use the provided launcher script:

./run_tui_app_docker.sh


Internally this runs:

docker run -it --rm --gpus all mytui:gpu
