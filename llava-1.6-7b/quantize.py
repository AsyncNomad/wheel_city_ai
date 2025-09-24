# quantize.py
# 7B 사이즈의 원본 모델을 돌리려면 20GB 이상의 VRAM이 필요함. 일반적인 사용자 환경에서는 불가능하므로 16비트로 표현된 가중치를 4비트로 양자화.
# 양자화 시 약 10%의 성능 손실이 있지만, VRAM 사용량이 약 1/4로 줄어 8GB의 VRAM 환경에서도 실행 가능.
import os
import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

# HF 리포 ID (원본은 캐시에서 자동 찾음)
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# 저장 위치
SAVE_DIR = os.path.join(".", "quantized", "llava-v1.6-mistral-7b-bnb4")
os.makedirs(SAVE_DIR, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Processor (토크나이저+이미지 프로세서)
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

# 모델 4bit 로드 (비전타워는 FP16 유지, 텍스트 가중치 4bit)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
)

# (선택) pad_token 세팅
if processor.tokenizer.pad_token_id is not None:
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

# 구성 저장(동일 설정으로 재로딩 가능)
processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR, safe_serialization=True)

print(f"[OK] Saved 4bit reloadable config to: {SAVE_DIR}")
