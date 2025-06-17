import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def tensor_to_image(tensor): #pyTorch 모델이 출력한 결과는 보통 tensor 형태(CxHxW)
    """Tensor [C, H, W] -> Numpy [H, W, C]""" # 근데 우리가 시각화하거나 PSNR 계산하려면 NumPy 이미지(HxWxC)가 필요
    image = tensor.detach().cpu().numpy() # GPU에 있는 tensor를 CPU로 옮기고 numpy로 바꿈
    image = np.transpose(image, (1, 2, 0))  # CxHxW -> HxWxC
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8) # [0~1] 범위를 0~255로 바꾸고 정수화
    return image

def image_to_tensor(image):
    """Numpy image -> Tensor [C, H, W]"""
    transform = transforms.ToTensor() # 역할: 일반 이미지를 PyTorch 모델이 사용할 수 있는 **Tensor(C, H, W)**로 바꿔줌
    return transform(image) # 이건 ToTensor()라는 torchvision의 기본 변환기로, 자동으로 255 스케일도 정규화해줘 (0~1 범위로)

def calculate_psnr(img1, img2):
    """PSNR 계산 (입력: numpy image)"""
    return peak_signal_noise_ratio(img1, img2, data_range=255) #역할: 복원된 이미지와 원본 이미지 간의 PSNR 계산
#PSNR (Peak Signal-to-Noise Ratio): 복원이 얼마나 원본에 가까운지를 수치로 측정
#값이 높을수록 좋음 (30dB 이상이면 괜찮은 품질)

def calculate_ssim(img1, img2):
    """SSIM 계산 (입력: numpy image)"""
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)
#역할: 복원 품질의 구조적 유사성을 계산
#SSIM (Structural Similarity Index): 사람 눈의 인식과 유사한 방식으로 복원 품질 측정
#1.0에 가까울수록 좋음

def resize_image(image, size):
    """OpenCV를 이용한 이미지 resize"""
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

#역할: 이미지 크기를 지정한 size로 리사이징 (예: (128, 128))

#INTER_CUBIC: 부드러운 고급 리사이징 방법 (4x4 주변 픽셀 사용해서 보간)


# ...뭔소린지 다는 못알아 먹음 뇌 터질거같고 나는 왜 이걸하고잇냐