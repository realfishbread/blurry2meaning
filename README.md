# 🚀 Super Resolution + AI 인식 실험 (FSRCNN + CLIP)

## 📌 프로젝트 개요

이 프로젝트는 저해상도 이미지를 딥러닝 기반 Super Resolution 모델을 통해 복원하고,  
복원된 이미지를 OpenAI의 CLIP 모델을 이용해 **"이 이미지가 무엇을 나타내는지" 자연어로 추론**하는 실험입니다.

- **입력**: 흐릿하고 해상도가 낮은 이미지 (예: 32x32)
- **복원 모델**: FSRCNN (Fast Super-Resolution CNN)
- **의미 추론 모델**: CLIP (Contrastive Language-Image Pretraining)

> 🌟 목표: AI가 인식하지 못하던 이미지를, 복원을 통해 **"의미를 이해할 수 있는 상태로 되돌리는 것"**

---

## 🧠 사용 기술 스택

- Python 3.x
- PyTorch
- torchvision
- HuggingFace `transformers` (CLIP)
- PIL, Matplotlib

---

## 🧪 실험 구성

1. **사용자가 흐릿한 이미지를 업로드**

   - 예: 빛이 거의 없는 자동차 사진

2. **FSRCNN 모델을 통해 복원**

   - 입력: 저해상도 이미지
   - 출력: 선명해진 이미지

3. **CLIP 모델로 의미 추론**
   - 텍스트 후보 중 가장 유사한 설명 추출 (ex: `"a blurry car"`)

---

## 💡 주요 결과 예시

| Original (입력)                    | Restored (복원 결과)             | CLIP 추론 결과            |
| ---------------------------------- | -------------------------------- | ------------------------- |
| ![](results/user_input_lowres.jpg) | ![](results/restored_output.jpg) | `"a blurry car"` (34.04%) |

> ✅ CLIP은 복원 전에는 인식이 어려웠던 이미지를,  
> 복원 후에는 **"자동차"로 추론 가능할 정도로 선명해졌음을 보여줌**

---

## 🏗️ 프로젝트 구조
