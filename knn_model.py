import os
import numpy as np
from PIL import Image
import joblib

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

MODEL_PATH = os.path.join('models', 'knn_rgb.pkl')

# Load model KNN sekali
KNN_MODEL = joblib.load(MODEL_PATH)

# HARUS sama dengan CLASSES di train_knn.py
LABELS = ['mangga', 'lain']   # atau ['mangga', 'jambu'] kalau foldermu begitu

def klasifikasi_dan_segmentasi(pil_image: Image.Image):
    """
    1) PIL.Image -> numpy RGB
    2) Segmentasi buah: mask & citra hasil segmentasi
    3) Ekstraksi ciri RGB
    4) Prediksi kelas dengan KNN
    """
    img_rgb = pil_image.convert('RGB')
    img_np = np.array(img_rgb)

    # 1) Segmentasi
    mask_np, img_seg_np = segment_fruit(img_np)

    # 2) Ekstraksi ciri RGB
    fitur_rgb = ekstrak_fitur_rgb(img_seg_np, mask_np)

    # 3) Prediksi kelas
    fitur_reshape = fitur_rgb.reshape(1, -1)
    pred_label = KNN_MODEL.predict(fitur_reshape)[0]

    return img_np, img_seg_np, mask_np, fitur_rgb, pred_label