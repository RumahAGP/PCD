# utils/segmentasi.py
import cv2
import numpy as np

def segment_fruit(img_rgb):
    """
    Segmentasi sederhana:
      - RGB -> grayscale
      - threshold Otsu + inverse (diasumsikan background lebih terang)
      - hasil: mask buah (255 = buah, 0 = background)
      - citra RGB hasil segmentasi (buah saja)
    Input:
      - img_rgb : array (H, W, 3) RGB, uint8
    Output:
      - mask    : array (H, W) 0/255
      - img_seg : array (H, W, 3) RGB, uint8
    """
    # RGB -> grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Threshold Otsu + inverse: objek gelap di background terang
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Bersihkan noise kecil (morfologi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Terapkan mask ke citra RGB
    img_seg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return mask, img_seg