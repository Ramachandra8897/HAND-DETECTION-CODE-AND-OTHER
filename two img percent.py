import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images_ssim(image1_path, image2_path):
    # Load images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute SSIM
    score, _ = ssim(img1, img2, full=True)
    percentage = score * 100  # Convert to percentage

    print(f"Image Similarity (SSIM): {percentage:.2f}%")
    return percentage

# Example usage
compare_images_ssim("C:\\Users\\Admin\\Pictures\\New folder (2)\\New folder (4)\\1737984588179.jpg", "c:\\Users\\Admin\\Pictures\\New folder (2)\\New folder (4)\\1737984588334.jpg")




