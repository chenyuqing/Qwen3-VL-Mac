# Qwen3-VL-2B-Instruct Local Deployment on MacOS

This project provides a **FastAPI-based Web UI** for running the **Qwen3-VL-2B-Instruct** multimodal model locally on Apple Silicon (M1/M2/M3/M4) Macs.

It features a modern chat interface, real-time streaming responses, and optimizations for Mac hardware (MPS acceleration).

## Features

- **🚀 Apple Silicon Optimized**: Runs efficiently on MPS (Metal Performance Shaders) with `bfloat16` precision.
- **⚡️ Streaming Responses**: Real-time token generation for instant feedback.
- **🎨 Modern Web UI**: iOS-inspired chat interface with light mode.
- **🖼️ Image Features**:
    - Drag & drop uploads.
    - Smart resizing to prevent OOM (Out Of Memory) errors.
    - **Lightbox Preview**: Click any image to view in full screen.
- **🛠️ Robust Server**:
    - Concurrency protection (Thread Locking).
    - Unique file naming (UUID).
    - Automatic memory caching cleanup.
    - End-to-end timeout protection.

## Prerequisites

- **Mac with Apple Silicon** (M1 or later).
- **Python 3.10+** (Recommended: Use `uv` for management).
- **RAM**: 16GB unified memory recommended.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/chenyuqing/Qwen3-VL-Mac.git
    cd Qwen3-VL-Mac
    ```

2.  **Initialize Environment**:
    We recommend using `uv` for fast dependency management.
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    uv pip install -r requirements_web_demo.txt
    ```
    *Note: `requirements_web_demo.txt` should include `fastapi`, `uvicorn`, `python-multipart`, `transformers`, `torch`, `qwen_vl_utils`, `accelerate`, etc.*

4.  **Download the Model**:
    The server expects the model to be locally cached or downloaded. You can download it using `huggingface-cli`:
    ```bash
    huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
    ```
    *By default, the `server.py` attempts to find the model in your standard HuggingFace cache. You may need to update `MODEL_PATH` in `server.py` if you store it elsewhere.*

## Usage

### Starting the Server

Use the provided helper script to start the server in the background:

```bash
./start.sh
```

- Local URL: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Logs: `tail -f server.log`

### Stopping the Server

```bash
./stop.sh
```

## Directory Structure

- `server.py`: The main FastAPI backend handling model inference and streaming.
- `static/index.html`: The HTML/JS frontend.
- `start.sh` / `stop.sh`: Service management scripts.
- `uploads/`: Temporary storage for uploaded images.

## Configuration

You can modify `server.py` to adjust:
- **`MODEL_PATH`**: Path to your local model.
- **`max_new_tokens`**: Maximum length of generated text (Default: 1024).

---
*Created for local AI experimentation.*
