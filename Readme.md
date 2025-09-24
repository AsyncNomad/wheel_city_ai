# 휠체어 접근성 분석 AI - Wheel City AI

`Wheel City AI`는 건물 입구 이미지 한 장으로 휠체어 접근성을 자동으로 분석하고 판단하는 딥러닝 기반 프로젝트입니다. **YOLOv8**의 객체 탐지 기술과 **LLaVA-1.6** 거대 언어 모델(VLM)의 상황 인지 능력을 결합하여, 사진 속 장소에 대해 이동 약자의 통행 가능 여부를 판단합니다.

## 프로젝트 메커니즘

이 프로젝트는 두 가지 AI 모델이 유기적으로 협력하는 파이프라인 구조로 동작합니다.

1. **1단계: 객체 탐지 (YOLOv8)**
    - 사용자가 `test_images` 폴더에 이미지를 입력하면, 사전 학습된 **YOLOv8s 모델**이 먼저 작동합니다.
    - 모델은 이미지 내에서 휠체어 접근성의 핵심 요소인 턱/계단(curb)과 경사로(ramp)를 탐지합니다.
    - 탐지된 객체에는 바운딩 박스(Bounding Box)가 표시되며, 이 시각화된 이미지는 LLaVA-1.6에 input으로 전달됩니다.
2. **2단계: 종합 판단 (LLaVA-1.6)**
    - 1단계에서 생성된 바운딩 박스 이미지를 **LLaVA-1.6 모델**이 입력받습니다.
    - LLaVA는 단순 객체 유무를 넘어, "턱이 있지만, 문으로 이어지는 유효한 경사로가 있는가?"와 같이 **이미지의 전체적인 맥락과 상황을 종합적으로 이해**하고 추론합니다.
    - 최종적으로, LLaVA는 접근성 규칙에 기반하여 `accessible` (접근 가능 여부), `reason` (판단 이유)이 포함된 구조화된 **JSON 형식의 최종 결과**를 생성합니다.

---

## 사용 모델 (Models Used)

| 역할 | 모델 이름 | 상세 정보 |
| --- | --- | --- |
| **객체 탐지** | YOLOv8s | Ultralytics의 작고 빠른 객체 탐지 모델 |
| **종합 판단** | LLaVA-1.6-7B-bnb4 | `bitsandbytes`를 이용한 4비트 양자화 버전 |

---

## 디렉토리 구조

```bash
wheel_city_ai/
├── yolov8/                   # YOLOv8 모델을 위한 프로젝트 폴더
│   ├── test_images/          # 1. 사용자가 원본 이미지를 넣는 곳(YOLOv8's input)
│   ├── test_results/         # 2. YOLOv8의 분석 결과(이미지)가 저장되는 곳(YOLOv8's output)
│   ├── run_model.py          # YOLOv8 분석을 실행하는 스크립트
│   └── ...                   # (학습 데이터, 모델 가중치 등)
│
├── llava-1.6-7b/             # LLaVA 모델을 위한 프로젝트 폴더
│   ├── test_images/          # 3. 분석을 위한 이미지를 넣는 곳 (YOLOv8's output=LLaVA's input)
│   ├── test_results/         # 4. LLaVA의 최종 JSON 결과가 저장되는 곳
│   ├── run_model_analyze.py  # LLaVA 분석을 실행하는 스크립트
│   └── ...                   # (모델 가중치 등)
│
├── run_pipeline.sh           # (선택사항) 전체 파이프라인을 한번에 실행하는 셔ㄹ 스크립트
└── README.md                 # 프로젝트 설명서 (현재 파일)
```

---

## 사용 방법

1. **이미지 입력:**
    - `yolov8/test_images/` 폴더에 분석하고 싶은 건물 입구 이미지를 넣습니다.
2. **YOLOv8 실행 (객체 탐지):**
    - `yolov8` 폴더로 이동하여 `run_model.py` 스크립트를 실행합니다.
    - 실행이 완료되면 `yolov8/test_results/visualizations/` 폴더에 바운딩 박스가 표시된 이미지들이 생성됩니다.
3. **LLaVA 실행 (최종 판단):**
    - `yolov8/test_results/visualizations/` 폴더에 있는 결과 이미지들을 `llava-1.6-7b/test_images/` 폴더로 **복사**합니다.
    - `llava-1.6-7b` 폴더로 이동하여 `run_model_analyze.py` 스크립트를 실행합니다.
4. **결과 확인:**
    - 최종 분석 결과는 `llava-1.6-7b/test_results/` 폴더 안의 `results.json` 파일에서 확인할 수 있습니다.

---

## 실행 예시

1. input 이미지 준비

 <img src="https://github.com/user-attachments/assets/58d210dc-d75a-4fa5-b0d0-7a3e4660ed41" width="400" height="600"/>

2. YOLOv8s가 턱과 경사로를 감지하여 핀

 <img src="https://github.com/user-attachments/assets/f514953d-f931-42ca-84d9-ed8f292fa24a" width="400" height="600"/>

3. LLaVA가 상황을 판단하여 최종 의사결정, 스크립트를 통해 JSON으로 파싱

```json
[
  {
    "image": "test_images/annotated_data1.jpg",
    "accessible": true,
    "reason": "Although a curve exists between the ground and the entrance to the store, a wheelchair can pass through through the ramp."
  }
]
```

---

## 🛠️ 환경 설정 (Ubuntu 24.04 기준)

### 1. 기본 시스템 패키지 설치

터미널을 열고 다음 명령어를 실행하여 Python과 가상환경 도구를 설치합니다.

```bash
sudo apt update
sudo apt install python3.10-venv python3-pip -y
```

### 2. NVIDIA 드라이버 및 CUDA 설치

LLaVA 모델의 4비트 양자화를 위해서는 NVIDIA GPU와 CUDA 환경이 필수적입니다.

- **NVIDIA 드라이버:** '소프트웨어 & 업데이트' 앱을 실행하여 '추가 드라이버' 탭에서 권장하는 NVIDIA 드라이버를 설치하세요.
- **CUDA Toolkit:** [NVIDIA CUDA Toolkit 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에서 Ubuntu 22.04 버전을 선택하여 설치를 진행합니다. (Ubuntu 24.04에서도 대부분 호환됩니다.)

### 3. 프로젝트 및 Python 패키지 설치

1. Git 리포지토리 복제: Bash
    
    ```bash
    git clone ~~
    cd wheel_city_ai
    ```
    
2. **각 프로젝트별 가상환경 생성 및 패키지 설치:**
    - **YOLOv8 환경 설정:**
        
        ```bash
        cd yolov8
        python3 -m venv .venv
        source .venv/bin/activate
        pip install ultralytics scikit-learn opencv-python
        ```
        
    - **LLaVA 환경 설정:**
        
        ```bash
        cd llava-1.6-7b
        python3 -m venv .venv
        source .venv/bin/activate
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install transformers bitsandbytes accelerate Pillow
        ```
        

### 4. Huggingface에서 LLaVA 다운로드, 양자화

1. LLaVA-1.6-7B 모델 로컬에 설치
    
    ```bash
    huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf
    ```
    
2. bitsandbytes 4bit 양자화 스크립트 실행
    
    ```bash
    cd llava-1.6-7b
    python3 -m venv .venv
    source .venv/bin/activate
    python quantize.py
    ```
    
    자동으로 llava-1.6-7b/quantized 디렉토리가 생성되고 양자화된 모델 정보가 저장됩니다. 이제 전체 모델을 실행할 준비를 마쳤습니다.
    

---

## 향후 계획

- 현재는 YOLOv8과 LLaVA-1.6-7B가 통합되어 하나의 유기적인 모델은 아님. 하나씩 돌려줘야 하지만 업데이트를 통해 두 모델을 통합할 예정임.
- 모델 학습에 필요한 데이터가 부족하고 아직 정확도가 떨어짐. 더 많은 학습 필요.