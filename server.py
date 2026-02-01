import os
import io
import time
import torch
import uuid
from threading import Thread, Lock
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from PIL import Image
from qwen_vl_utils import process_vision_info

app = FastAPI()

# Global Lock for GPU resources
gpu_lock = Lock()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Path and Configuration
MODEL_PATH = "/Users/tim/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH} to {DEVICE}...")

# Load Model and Processor
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

@app.post("/chat")
async def chat(prompt: str = Form(...), image: UploadFile = File(None)):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # 1. Read image bytes in main thread (async I/O)
    image_bytes = None
    image_name = None
    if image:
        image_bytes = await image.read()
        image_name = image.filename

    # 2. Setup Streamer
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    
    # 3. Define the background task
    def background_inference(prompt_text, img_bytes, img_name, stream_obj):
        try:
            # --- Preparation Phase (inside thread) ---
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
            
            if img_bytes:
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Smart Resize
                max_size = 1024
                if max(pil_image.size) > max_size:
                    pil_image.thumbnail((max_size, max_size))
                
                # Unique filename
                unique_name = f"upload_{uuid.uuid4()}.png"
                file_path = f"uploads/{unique_name}"
                pil_image.save(file_path)
                messages[0]["content"].insert(0, {"type": "image", "image": os.path.abspath(file_path)})

            # Tokenization
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            
            # --- Generation Phase (Locked) ---
            generation_kwargs = dict(inputs, max_new_tokens=1024, do_sample=False, streamer=stream_obj)
            
            with gpu_lock:
                try:
                    model.generate(**generation_kwargs)
                finally:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
        except Exception as e:
            # Send error to the stream
            stream_obj.put(f"[Error: {str(e)}]")
            # Important: end the stream so it doesn't hang
            stream_obj.end()
            import traceback
            traceback.print_exc()

    # 4. Start Thread
    thread = Thread(target=background_inference, args=(prompt, image_bytes, image_name, streamer))
    thread.start()

    # 5. Return Stream
    async def response_generator():
        try:
            for new_text in streamer:
                yield new_text
        except Exception as e:
            yield f"[Stream Error: {str(e)}]"

    return StreamingResponse(response_generator(), media_type="text/plain")

# Serve Static Files (Frontend)
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
