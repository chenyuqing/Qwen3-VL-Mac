from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Use the local model path provided by the user
model_path = "/Users/tim/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

def run_inference():
    print(f"Loading model from {model_path}...")
    
    # We use Qwen2VLForConditionalGeneration as Qwen3-VL typically shares the architecture
    # or falls back to AutoModel if not specific. Qwen2VL is the safe bet for VLM if compatible.
    # Using 'auto' mapping is preferred but let's try explicit class if known or Auto.
    # Let's use AutoModel first to let transformers decide.
    from transformers import AutoModelForVision2Seq

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True 
        )
    except Exception as e:
        print(f"Failed to load with AutoModelForVision2Seq: {e}")
        print("Trying Qwen2VLForConditionalGeneration...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("Model loaded successfully.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
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

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\nInference Output:")
    print(output_text[0])

if __name__ == "__main__":
    run_inference()
