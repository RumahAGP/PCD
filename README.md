#!/bin/bash

echo "=== SISTEM KLASIFIKASI KNN ==="

echo "1. Menghapus folder models..."
rm -rf models
mkdir models
echo "   -> Folder models dibersihkan."

echo "2. Memulai proses training KNN..."
python train_knn.py

echo "   -> Training selesai."
echo "   -> File model tersimpan di folder models/"

echo "3. Menjalankan website (app.py)..."
python app.py
