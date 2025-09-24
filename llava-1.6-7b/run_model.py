# run_model.py
import os
import sys
import glob
import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

# Prefer a locally saved 4-bit directory first, else fall back to HF repo (loads in 4-bit)
LOCAL_MODEL_DIR = os.path.join(".", "quantized", "llava-v1.6-mistral-7b-bnb4")
HF_REPO_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_model():
    print("[INFO] Loading model...")
    model_src = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_REPO_ID
    processor = LlavaNextProcessor.from_pretrained(model_src)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_src,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # Optional pad token
    if processor.tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    print("[OK] Model loaded.")
    return processor, model

def is_image_file(path: str) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return os.path.isfile(path) and path.lower().endswith(exts)

def answer_image(processor, model, image_path: str, question: str, max_new_tokens: int = 256) -> str:
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    # LLaVA Next chat template
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Build inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # ✅ Move to device, keep correct dtypes (input_ids must stay Long/Int)
    moved = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            moved[k] = v
            continue
        if k in ("pixel_values", "images"):  # image tensors → fp16 is fine
            moved[k] = v.to(DEVICE, dtype=torch.float16)
        else:
            moved[k] = v.to(DEVICE)  # keep dtype (e.g., input_ids Long)
    inputs = moved

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    parts = text.split("assistant\n")
    return (parts[-1] if len(parts) > 1 else text).strip()

def main():
    processor, model = load_model()
    print("\n[READY] Enter an image file path and a question. Press Ctrl+C to exit.")

    while True:
        try:
            img_path = input("\nImage file path: ").strip()
            if not img_path:
                print("  - Empty path. Please try again.")
                continue
            if not os.path.exists(img_path):
                print(f"  - Path does not exist: {img_path}")
                continue
            if not is_image_file(img_path):
                print("  - Not an image file. Supported: .jpg .jpeg .png .webp .bmp")
                continue

            question = input("Question: ").strip()
            if not question:
                print("  - Empty question. Please try again.")
                continue

            print("[RUN] Generating answer...")
            try:
                ans = answer_image(processor, model, img_path, question)
                print(f"\n=== {os.path.basename(img_path)} ===")
                print(ans)
                print("\n[DONE] You can ask another question or choose another image.")
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")

        except KeyboardInterrupt:
            print("\n\n[EXIT] Stopped by user (Ctrl+C).")
            break
        except EOFError:
            print("\n\n[EXIT] End of input.")
            break

if __name__ == "__main__":
    main()
