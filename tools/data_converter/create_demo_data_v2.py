#!/usr/bin/env python
"""创建 KITTI 格式的小规模演示数据 (mmdet3d 兼容版本)

用于测试 PointPillars 训练流程，生成少量模拟点云和标注数据。
支持 mmdet3d 1.4.0 的数据格式。

Usage:
    python tools/data_converter/create_demo_data_v2.py --num-samples 50
"""

import argparse
import os
import os.path as osp
import pickle
from typing import List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create demo KITTI data")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/kitti",
        help="Output root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def generate_random_points(
    num_points: int = 20000,
    point_range: List[float] = [0, -40, -3, 70, 40, 1],
) -> np.ndarray:
    """Generate random point cloud within specified range.

    Args:
        num_points: Number of points to generate.
        point_range: [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Points array of shape (N, 4) with [x, y, z, intensity].
    """
    x = np.random.uniform(point_range[0], point_range[3], num_points)
    y = np.random.uniform(point_range[1], point_range[4], num_points)
    z = np.random.uniform(point_range[2], point_range[5], num_points)

    # Add ground plane (more points near z=0)
    ground_mask = np.random.rand(num_points) < 0.3
    z[ground_mask] = np.random.uniform(-0.5, 0.2, ground_mask.sum())

    # Random intensity
    intensity = np.random.uniform(0, 1, num_points)

    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    return points


def generate_random_boxes(
    num_boxes: int = 5,
    point_range: List[float] = [0, -40, -3, 70, 40, 1],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random 3D bounding boxes.

    Args:
        num_boxes: Number of boxes to generate.
        point_range: Valid range for box centers.

    Returns:
        boxes: (N, 7) array [x, y, z, l, w, h, rot] (mmdet3d format: l, w, h)
        labels: (N,) array of class labels
        names: list of class names
    """
    classes = ["Car", "Pedestrian", "Cyclist"]
    class_dims = {
        # mmdet3d uses (l, w, h) order
        "Car": ([3.9, 1.6, 1.56], 0.3),  # (mean [l,w,h], std)
        "Pedestrian": ([0.8, 0.6, 1.73], 0.15),
        "Cyclist": ([1.76, 0.6, 1.73], 0.2),
    }

    boxes = []
    labels = []
    names = []

    for _ in range(num_boxes):
        # Random class
        cls_idx = np.random.randint(0, 3)
        cls_name = classes[cls_idx]
        mean_dims, std = class_dims[cls_name]

        # Random center (avoid edges)
        x = np.random.uniform(point_range[0] + 5, point_range[3] - 5)
        y = np.random.uniform(point_range[1] + 5, point_range[4] - 5)
        z = np.random.uniform(-1, 0.5)

        # Random dimensions with some variation (l, w, h)
        l = max(0.5, mean_dims[0] + np.random.randn() * std)
        w = max(0.3, mean_dims[1] + np.random.randn() * std)
        h = max(0.5, mean_dims[2] + np.random.randn() * std)

        # Random rotation
        rot = np.random.uniform(-np.pi, np.pi)

        # mmdet3d format: [x, y, z, l, w, h, rot]
        boxes.append([x, y, z, l, w, h, rot])
        labels.append(cls_idx)
        names.append(cls_name)

    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return boxes, labels, names


def create_calib_file(calib_path: str):
    """Create a dummy calibration file."""
    # Standard KITTI calibration matrices
    P2 = np.array([
        721.5377, 0, 609.5593, 44.85728,
        0, 721.5377, 172.854, 0.2163791,
        0, 0, 1, 0.002745884
    ])
    R0_rect = np.eye(3).flatten()
    Tr_velo_to_cam = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
        1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
        9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01
    ])
    Tr_imu_to_velo = np.array([
        9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01,
        -7.854027e-04, 9.998898e-01, -1.482298e-02, 3.195559e-01,
        2.024406e-03, 1.482454e-02, 9.998881e-01, -7.997231e-01
    ])

    with open(calib_path, "w") as f:
        f.write(f"P0: {' '.join(map(str, P2))}\n")
        f.write(f"P1: {' '.join(map(str, P2))}\n")
        f.write(f"P2: {' '.join(map(str, P2))}\n")
        f.write(f"P3: {' '.join(map(str, P2))}\n")
        f.write(f"R0_rect: {' '.join(map(str, R0_rect))}\n")
        f.write(f"Tr_velo_to_cam: {' '.join(map(str, Tr_velo_to_cam))}\n")
        f.write(f"Tr_imu_to_velo: {' '.join(map(str, Tr_imu_to_velo))}\n")


def create_label_file(label_path: str, boxes: np.ndarray, names: List[str]):
    """Create a KITTI format label file.

    Note: We store LiDAR boxes, but KITTI labels are in camera coord.
    For simplicity, we use dummy camera-coord values.
    """
    with open(label_path, "w") as f:
        for i, (box, name) in enumerate(zip(boxes, names)):
            # Dummy 2D bbox
            bbox_2d = [100, 100, 200, 200]
            # Dummy camera location (approximate conversion)
            loc_cam = [box[0], -box[2], box[1]]  # rough x,y,z swap
            # Dimensions in camera coord (h, w, l)
            dims_cam = [box[5], box[4], box[3]]  # h, w, l
            # Rotation
            rot = -box[6] - np.pi / 2

            line = f"{name} 0.0 0 0.0 "
            line += f"{bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} "
            line += f"{dims_cam[0]:.2f} {dims_cam[1]:.2f} {dims_cam[2]:.2f} "
            line += f"{loc_cam[0]:.2f} {loc_cam[1]:.2f} {loc_cam[2]:.2f} "
            line += f"{rot:.2f}\n"
            f.write(line)


def create_info_pkl_mmdet3d(
    data_root: str,
    sample_ids: List[str],
    boxes_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    names_list: List[List[str]],
    out_path: str,
):
    """Create info pickle file in mmdet3d 1.4.0 format.
    
    mmdet3d 1.4.0 requires 'instances' field instead of 'annos'.
    """
    infos = []

    for idx, (sample_id, boxes, labels, names) in enumerate(
        zip(sample_ids, boxes_list, labels_list, names_list)
    ):
        # Build instances list (mmdet3d 1.4.0 format)
        instances = []
        for i, (box, label, name) in enumerate(zip(boxes, labels, names)):
            instance = {
                'bbox_3d': box.tolist(),  # [x, y, z, l, w, h, rot]
                'bbox_label_3d': int(label),
                # Camera bbox (dummy values for LiDAR-only)
                'bbox': [100, 100, 200, 200],
                'bbox_label': int(label),
                # Additional fields
                'truncated': 0.0,
                'occluded': 0,
                'alpha': 0.0,
                'score': 1.0,
                'difficulty': 1,  # moderate
                'num_lidar_pts': 100,  # dummy value
                'group_id': i,
            }
            instances.append(instance)

        info = {
            'sample_idx': int(sample_id),
            'lidar_points': {
                'lidar_path': f'training/velodyne_reduced/{sample_id}.bin',
                'num_pts_feats': 4,
            },
            'instances': instances,
            # Calibration matrices (dummy but valid)
            'calib': {
                'P2': np.array([
                    [721.5377, 0, 609.5593, 44.85728],
                    [0, 721.5377, 172.854, 0.2163791],
                    [0, 0, 1, 0.002745884],
                ]),
                'R0_rect': np.eye(3),
                'Tr_velo_to_cam': np.array([
                    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                ]),
            },
        }
        infos.append(info)

    data = {
        'metainfo': {
            'dataset': 'KITTI',
            'version': 'mmdet3d_1.4',
            'info_version': '1.1',
            'classes': ['Car', 'Pedestrian', 'Cyclist'],
        },
        'data_list': infos,
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_root = args.data_root
    num_samples = args.num_samples

    print("=" * 60)
    print("Creating Demo KITTI Data (mmdet3d 1.4.0 compatible)")
    print("=" * 60)
    print(f"Output directory: {data_root}")
    print(f"Number of samples: {num_samples}")
    print()

    # Create directories (including velodyne_reduced for mmdet3d)
    dirs = [
        osp.join(data_root, "training", "velodyne"),
        osp.join(data_root, "training", "velodyne_reduced"),
        osp.join(data_root, "training", "calib"),
        osp.join(data_root, "training", "label_2"),
        osp.join(data_root, "ImageSets"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Generate data
    sample_ids = []
    boxes_list = []
    labels_list = []
    names_list = []

    print("Generating samples...")
    for i in range(num_samples):
        sample_id = f"{i:06d}"
        sample_ids.append(sample_id)

        # Generate point cloud
        num_points = np.random.randint(15000, 25000)
        points = generate_random_points(num_points)
        
        # Save to both velodyne and velodyne_reduced
        points_path = osp.join(data_root, "training", "velodyne", f"{sample_id}.bin")
        points.tofile(points_path)
        points_path_reduced = osp.join(data_root, "training", "velodyne_reduced", f"{sample_id}.bin")
        points.tofile(points_path_reduced)

        # Generate boxes
        num_boxes = np.random.randint(3, 10)
        boxes, labels, names = generate_random_boxes(num_boxes)
        boxes_list.append(boxes)
        labels_list.append(labels)
        names_list.append(names)

        # Create calibration file
        calib_path = osp.join(data_root, "training", "calib", f"{sample_id}.txt")
        create_calib_file(calib_path)

        # Create label file
        label_path = osp.join(data_root, "training", "label_2", f"{sample_id}.txt")
        create_label_file(label_path, boxes, names)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    print()

    # Create train/val splits
    num_train = int(num_samples * 0.8)
    train_ids = sample_ids[:num_train]
    val_ids = sample_ids[num_train:]

    # Save ImageSets
    with open(osp.join(data_root, "ImageSets", "train.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(osp.join(data_root, "ImageSets", "val.txt"), "w") as f:
        f.write("\n".join(val_ids))
    with open(osp.join(data_root, "ImageSets", "trainval.txt"), "w") as f:
        f.write("\n".join(sample_ids))

    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    print()

    # Create info pkl files (mmdet3d format)
    print("Creating info pkl files (mmdet3d 1.4.0 format)...")

    create_info_pkl_mmdet3d(
        data_root,
        train_ids,
        boxes_list[:num_train],
        labels_list[:num_train],
        names_list[:num_train],
        osp.join(data_root, "kitti_infos_train.pkl"),
    )

    create_info_pkl_mmdet3d(
        data_root,
        val_ids,
        boxes_list[num_train:],
        labels_list[num_train:],
        names_list[num_train:],
        osp.join(data_root, "kitti_infos_val.pkl"),
    )

    print()
    print("=" * 60)
    print("Demo data created successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - {num_samples} point cloud files (.bin)")
    print(f"  - {num_samples} calibration files (.txt)")
    print(f"  - {num_samples} label files (.txt)")
    print(f"  - kitti_infos_train.pkl ({len(train_ids)} samples)")
    print(f"  - kitti_infos_val.pkl ({len(val_ids)} samples)")
    print()
    print("You can now run training with:")
    print("  # Using mmdet3d components:")
    print("  python tools/train.py configs/pointpillars_mmdet3d/pointpillars_kitti.py")
    print()
    print("  # Using custom mmlite components:")
    print("  python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6_kitti-3d-3class.py")


if __name__ == "__main__":
    main()
