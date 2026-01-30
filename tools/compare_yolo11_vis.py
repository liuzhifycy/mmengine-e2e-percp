
import os
import sys

# Add project root and local ultralytics to path
project_root = os.getcwd()
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'ultralytics'))
sys.path.append(os.path.join(project_root, 'mmengine-lite'))  # Add path to find mmlite package

import cv2
import torch
import numpy as np
import random
from ultralytics import YOLO
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

# Ensure mmlite modules are registered
register_all_modules()
import mmlite.models  # Import to register mmlite models explicitly

def run_comparison():
    # Paths
    work_dir = "mmengine-lite"
    data_dir = os.path.join(work_dir, "data/coco/val2017")
    output_dir = os.path.join(work_dir, "vis_comparison")
    ultra_weights = os.path.join(work_dir, "yolo11m.pt")
    mm_config = os.path.join(work_dir, "configs/yolo11/yolo11m_coco_train.py")
    mm_weights = os.path.join(work_dir, "yolo11m_mm.pth")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Ultralytics Model
    print(f"Loading Ultralytics model: {ultra_weights}")
    ultra_model = YOLO(ultra_weights)

    # 2. Load MMEngine Model
    print(f"Loading MMEngine model: {mm_config}")
    mm_model = init_detector(mm_config, mm_weights, device='cuda:0')

    # 3. Select Images
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("No images found in data directory!")
        return
    
    selected_images = random.sample(image_files, 5)
    print(f"Selected images: {selected_images}")

    for img_name in selected_images:
        img_path = os.path.join(data_dir, img_name)
        print(f"\nProcessing {img_name}...")
        
        # --- Ultralytics Inference ---
        # conf=0.25 to match typical mmdet defaults or set explicitly
        ultra_results = ultra_model(img_path, conf=0.3, iou=0.45, verbose=False)[0]
        ultra_vis = ultra_results.plot() # Returns BGR numpy array

        # --- MMEngine Inference ---
        mm_result = inference_detector(mm_model, img_path)
        
        # Visualization for MM
        visualizer = VISUALIZERS.build(mm_model.cfg.visualizer)
        visualizer.dataset_meta = mm_model.dataset_meta
        
        # Draw
        img = cv2.imread(img_path)
        img = img[:, :, ::-1] # BGR to RGB for visualizer? No, visualizer handles it usually, but let's check.
        # MMDet visualizer usually takes RGB
        
        visualizer.add_datasample(
            'result',
            img,
            data_sample=mm_result,
            draw_gt=False,
            wait_time=0,
            pred_score_thr=0.3
        )
        mm_vis = visualizer.get_image()
        # Convert RGB to BGR for OpenCV
        mm_vis = mm_vis[:, :, ::-1]

        # --- Comparison & Saving ---
        # Resize to same height if needed (usually same since same input, but ensure)
        h1, w1 = ultra_vis.shape[:2]
        h2, w2 = mm_vis.shape[:2]
        
        if h1 != h2:
            mm_vis = cv2.resize(mm_vis, (int(w2 * h1 / h2), h1))
        
        # Concatenate horizontally
        combined = np.concatenate((ultra_vis, mm_vis), axis=1)
        
        # Add labels
        cv2.putText(combined, "Ultralytics", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "MMEngine-Lite", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out_path = os.path.join(output_dir, f"compare_{img_name}")
        cv2.imwrite(out_path, combined)
        print(f"Saved comparison to {out_path}")

        # --- Textual Comparison (Top 5 boxes) ---
        print("  > Ultralytics Detections (Top 5):")
        if ultra_results.boxes:
            boxes = ultra_results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
            # Sort by conf
            boxes = boxes[boxes[:, 4].argsort()[::-1]]
            for i in range(min(5, len(boxes))):
                b = boxes[i]
                print(f"    Class: {int(b[5])}, Conf: {b[4]:.4f}, Box: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")
        
        print("  > MMEngine Detections (Top 5):")
        pred_instances = mm_result.pred_instances
        scores = pred_instances.scores
        labels = pred_instances.labels
        bboxes = pred_instances.bboxes
        
        # Filter by score thr
        mask = scores > 0.3
        scores = scores[mask]
        labels = labels[mask]
        bboxes = bboxes[mask]
        
        if len(scores) > 0:
            # Sort
            indices = torch.argsort(scores, descending=True)
            for i in range(min(5, len(indices))):
                idx = indices[i]
                print(f"    Class: {labels[idx]}, Conf: {scores[idx]:.4f}, Box: [{bboxes[idx][0]:.1f}, {bboxes[idx][1]:.1f}, {bboxes[idx][2]:.1f}, {bboxes[idx][3]:.1f}]")
        else:
            print("    No detections > 0.3")

if __name__ == "__main__":
    import mmengine
    run_comparison()
