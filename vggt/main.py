import os
import cv2
import torch
import open3d as o3d
import numpy as np
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

# === Параметри ===
video_path = "videos/fern.mp4"
frame_output_dir = "imgs"
frame_limit = 4
batch_size = 2

os.makedirs(frame_output_dir, exist_ok=True)

# === 1. Витягуємо кадри з відео ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
print("🎞️ Витягуємо кадри з відео...")

while cap.isOpened() and frame_count < frame_limit:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_path = os.path.join(frame_output_dir, f"frame_{frame_count}.png")
    Image.fromarray(frame_rgb).save(img_path)
    frame_count += 1
cap.release()
print(f"✅ Збережено {frame_count} кадрів у {frame_output_dir}/")

# === 2. Завантаження шляхів до кадрів ===
image_paths = sorted([
    os.path.join(frame_output_dir, f)
    for f in os.listdir(frame_output_dir)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# === 3. Обробка batch-wise ===
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    print(f"🔄 Обробка кадрів {i}–{i+len(batch_paths)-1}...")
    
    images = load_and_preprocess_images(batch_paths).to(device)

    with torch.no_grad():
        predictions = model(images)

    print("✅ Інференс завершено")
    print("🔑 Ключі:", predictions.keys())

    # === Витягуємо потрібне ===
    if all(k in predictions for k in ['point_map', 'depth_map', 'intrinsics', 'extrinsics']):
        point_map = predictions["point_map"].cpu().numpy()         # (B, H, W, 3)
        depth_map = predictions["depth_map"].cpu().numpy()         # (B, H, W)
        intrinsics = predictions["intrinsics"].cpu().numpy()       # (B, 3, 3)
        extrinsics = predictions["extrinsics"].cpu().numpy()       # (B, 4, 4)

        print(f"📌 point_map: {point_map.shape}")
        print(f"📌 depth_map: {depth_map.shape}")
        print(f"📌 intrinsics[0]:\n{intrinsics[0]}")
        print(f"📌 extrinsics[0]:\n{extrinsics[0]}")

        # === Візуалізація point_map ===
        flat_points = point_map.reshape(-1, 3)
        flat_points = flat_points[~np.isnan(flat_points).any(axis=1)]  # фільтруємо nan
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(flat_points)
        o3d.visualization.draw_geometries([pcd], window_name="VGGT — 3D point_map")

    else:
        print("⚠️ Один із ключів відсутній. Спробуй інші кадри або відео.")
