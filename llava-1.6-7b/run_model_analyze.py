import os, re, json, glob, argparse, torch, traceback
from PIL import Image
from transformers import (BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration)

LOCAL_MODEL_DIR = os.path.join(".", "quantized", "llava-v1.6-mistral-7b-bnb4")
HF_REPO_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".bmp")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

PROMPT = (
    "You are an accessibility analysis AI. Analyze the provided image of a building entrance to determine if it is accessible for a lone wheelchair user.\n"
    "Accessibility Rules:\n"
    "1. There must be no steps or curbs between the ground and the entrance.\n"
    "2. If there are steps or curbs, a permanent ramp must connect the ground to the entrance.\n\n"
    "Respond with ONLY a single, valid JSON object. "
    "The 'accessible' field must be a boolean (true/false). Output null only in inevitable cases where no amount of hard work can be done to determine.\n"
    'The JSON schema is: {"accessible": boolean | null, "reason": string}'
)
SYSTEM = "You are an AI assistant that only outputs JSON."


def load_model():
    src = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_REPO_ID
    print(f"[INFO] Loading model from: {src}")
    proc = LlavaNextProcessor.from_pretrained(src)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        src, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto"
    )
    return proc, model

def extract_json(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: return None
    
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str_fixed = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            return None

def infer_one(proc, model, img_path):
    image = Image.open(img_path).convert("RGB")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    prompt_chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt_chat, images=image, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, temperature=0.0, max_new_tokens=150)
    
    full_text = proc.batch_decode(out, skip_special_tokens=True)[0]
    
    assistant_response = full_text.split("[/INST]")[-1].strip()

    # (디버깅용) 원본 답변을 터미널에 출력
    print(f"\n[DEBUG] Raw model output for {os.path.basename(img_path)}:\n---\n{assistant_response}\n---\n")

    obj = extract_json(assistant_response) or {"accessible": None, "reason": "invalid model output"}
    
    if "reason" not in obj or not isinstance(obj.get("reason"), str):
        obj["reason"] = str(obj.get("reason", ""))
    obj["reason"] = obj["reason"].strip()[:50]
    
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="./test_images")
    ap.add_argument("--output_dir", default="./test_results")
    ap.add_argument("--outfile", default="results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    proc, model = load_model()

    imgs = [p for ext in IMAGE_EXTS for p in glob.glob(os.path.join(args.input_dir, f"*{ext}"))]
    imgs.sort()
    
    if not imgs:
        print(f"[WARN] No images found in {args.input_dir}")
        return

    results = []
    for p in imgs:
        print(f"[RUN] {os.path.basename(p)}")
        try:
            obj = infer_one(proc, model, p)
        except Exception:
            error_details = traceback.format_exc()
            print(f"[ERROR] An exception occurred while processing {os.path.basename(p)}:")
            print(error_details)
            obj = {"accessible": None, "reason": f"Exception caught: {error_details}"}

        results.append({
            "image": os.path.relpath(p),
            "accessible": obj.get("accessible"),
            "reason": obj.get("reason")
        })

    out_path = os.path.join(args.output_dir, f"{args.outfile}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Wrote {len(results)} items to {out_path}")

if __name__ == "__main__":
    main()