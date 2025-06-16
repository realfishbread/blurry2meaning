import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def tensor_to_image(tensor): #pyTorch 모델이 출력한 결과는 보통 tensor 형태(CxHxW)
    """Tensor [C, H, W] -> Numpy [H, W, C]""" # 근데 우리가 시각화하거나 PSNR 계산하려면 NumPy 이미지(HxWxC)가 필요
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # CxHxW -> HxWxC
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

def image_to_tensor(image):
    """Numpy image -> Tensor [C, H, W]"""
    transform = transforms.ToTensor()
    return transform(image)

def calculate_psnr(img1, img2):
    """PSNR 계산 (입력: numpy image)"""
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """SSIM 계산 (입력: numpy image)"""
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)

def resize_image(image, size):
    """OpenCV를 이용한 이미지 resize"""
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)