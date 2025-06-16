# 🚀 Super Resolution + AI 인식 실험 (FSRCNN + ResNet18)

## 📌 프로젝트 개요

이 프로젝트는 저해상도 이미지를 딥러닝 기반 Super Resolution 모델을 통해 복원하고,  
복원된 이미지를 ResNet18 분류 모델을 이용해 **"이 이미지가 무엇을 나타내는지" 자연어로 추론**하는 실험입니다.

- **입력**: 흐릿하고 해상도가 낮은 이미지 (예: 32x32)
- **복원 모델**: FSRCNN (Fast Super-Resolution CNN)
- **의미 추론 모델**: ResNet18

> 🌟 목표: AI가 인식하지 못하던 이미지를, 복원을 통해 **"의미를 이해할 수 있는 상태로 되돌리는 것"**

---

## 🧠 사용 기술 스택

- Python 3.x
- PyTorch
- torchvision (ResNet18)
- PIL, Matplotlib

---

## 🧪 실험 구성

1. **사용자가 흐릿한 이미지를 업로드**

   - 예: 빛이 거의 없는 자동차 사진

2. **FSRCNN 모델을 통해 복원**

   - 입력: 저해상도 이미지
   - 출력: 선명해진 이미지

3. **ResNet18 모델로 의미 추론**
   - 텍스트 후보 중 가장 유사한 설명 추출 (ex: `"a blurry car"`)

---

## 📊 복원 + AI 추론 결과 (확률 포함)

| 입력 이미지 | 복원 이미지 | AI 분류 결과 | 신뢰도 (%) | 해석 |
|--------------|--------------|----------------|--------------|--------|
| ![](./data/low_res/ad8e0cd7-f025-4ce5-be03-4f3bd118e5f2.jpg) | ![](./results/predicted/ad8e0cd7-f025-4ce5-be03-4f3bd118e5f2.jpg) | **vase** | 77.19% | 분류 성공 |
| ![](./data/low_res/images%20(1).jpg) | ![](./results/predicted/images%20(1).jpg) | **lighter** | 12.78% | 분류 실패 가능성 |
| ![](./data/low_res/images.jpg) | ![](./results/predicted/images.jpg) | **Labrador retriever** | 54.14% | 다소 애매하지만 유사 |

