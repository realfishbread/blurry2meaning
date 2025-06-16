import os
import cv2
from utils.image_utils import calculate_psnr, calculate_ssim

pred_folder = "results/predicted"
gt_folder = "data/high_res"

print("\n📊 복원 성능 평가 결과 (PSNR / SSIM):\n")
total_psnr = 0
total_ssim = 0
count = 0

for filename in os.listdir(pred_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    pred_path = os.path.join(pred_folder, filename)
    gt_path = os.path.join(gt_folder, filename)

    if not os.path.exists(gt_path):
        print(f"❌ GT 없음: {filename}")
        continue

    pred_img = cv2.imread(pred_path)
    gt_img = cv2.imread(gt_path)

    if pred_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

    psnr = calculate_psnr(gt_img, pred_img)
    ssim = calculate_ssim(gt_img, pred_img)

    print(f"{filename}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    total_psnr += psnr
    total_ssim += ssim
    count += 1

if count > 0:
    print("\n📈 평균 PSNR: {:.2f}".format(total_psnr / count))
    print("📈 평균 SSIM: {:.4f}".format(total_ssim / count))
else:
    print("⚠️ 평가할 이미지가 없습니다.")
