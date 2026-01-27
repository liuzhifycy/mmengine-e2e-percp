#!/usr/bin/env python
"""KITTI 数据转换器 - 生成 mmengine-lite 格式的 .pkl 标注文件

将 KITTI 原始数据转换为训练所需的 pickle 格式标注文件。

Usage:
    python tools/data_converter/kitti_converter.py --data-root data/kitti

输出文件:
    - kitti_infos_train.pkl
    - kitti_infos_val.pkl
    - kitti_infos_trainval.pkl
    - kitti_infos_test.pkl
"""

import argparse
import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="KITTI Data Converter")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/kitti",
        help="Root path of KITTI dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for pkl files (default: same as data-root)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    return parser.parse_args()


def read_calib(calib_path: str) -> Dict[str, np.ndarray]:
    """Read KITTI calibration file.

    Args:
        calib_path: Path to calibration file.

    Returns:
        Dict containing calibration matrices.
    """
    calib = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            calib[key.strip()] = np.array(
                [float(x) for x in value.strip().split()], dtype=np.float32
            )

    # Parse matrices
    P0 = calib.get("P0", np.zeros(12)).reshape(3, 4)
    P1 = calib.get("P1", np.zeros(12)).reshape(3, 4)
    P2 = calib.get("P2", np.zeros(12)).reshape(3, 4)
    P3 = calib.get("P3", np.zeros(12)).reshape(3, 4)
    R0_rect = np.eye(4, dtype=np.float32)
    R0_rect[:3, :3] = calib.get("R0_rect", np.eye(9)).reshape(3, 3)
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3, :4] = calib.get("Tr_velo_to_cam", np.zeros(12)).reshape(3, 4)
    Tr_imu_to_velo = np.eye(4, dtype=np.float32)
    Tr_imu_to_velo[:3, :4] = calib.get("Tr_imu_to_velo", np.zeros(12)).reshape(3, 4)

    return {
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam,
        "Tr_imu_to_velo": Tr_imu_to_velo,
    }


def read_label(label_path: str) -> Dict[str, np.ndarray]:
    """Read KITTI label file.

    Args:
        label_path: Path to label file.

    Returns:
        Dict containing annotation arrays.
    """
    if not osp.exists(label_path):
        return {
            "name": np.array([]),
            "truncated": np.array([]),
            "occluded": np.array([]),
            "alpha": np.array([]),
            "bbox": np.zeros((0, 4)),
            "dimensions": np.zeros((0, 3)),
            "location": np.zeros((0, 3)),
            "rotation_y": np.array([]),
            "score": np.array([]),
            "difficulty": np.array([]),
        }

    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return {
            "name": np.array([]),
            "truncated": np.array([]),
            "occluded": np.array([]),
            "alpha": np.array([]),
            "bbox": np.zeros((0, 4)),
            "dimensions": np.zeros((0, 3)),
            "location": np.zeros((0, 3)),
            "rotation_y": np.array([]),
            "score": np.array([]),
            "difficulty": np.array([]),
        }

    names = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dimensions = []
    location = []
    rotation_y = []
    score = []
    difficulty = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        names.append(parts[0])
        truncated.append(float(parts[1]))
        occluded.append(int(parts[2]))
        alpha.append(float(parts[3]))
        bbox.append([float(x) for x in parts[4:8]])
        # KITTI format: height, width, length -> we store as h, w, l
        dimensions.append([float(parts[8]), float(parts[9]), float(parts[10])])
        location.append([float(x) for x in parts[11:14]])
        rotation_y.append(float(parts[14]))

        if len(parts) >= 16:
            score.append(float(parts[15]))
        else:
            score.append(1.0)

        # Compute difficulty
        # Easy: bbox height > 40, occlusion = 0, truncation < 0.15
        # Moderate: bbox height > 25, occlusion <= 1, truncation < 0.3
        # Hard: bbox height > 25, occlusion <= 2, truncation < 0.5
        h = float(parts[7]) - float(parts[5])  # bbox height
        trunc = float(parts[1])
        occ = int(parts[2])

        if h > 40 and occ == 0 and trunc < 0.15:
            diff = 0  # Easy
        elif h > 25 and occ <= 1 and trunc < 0.3:
            diff = 1  # Moderate
        elif h > 25 and occ <= 2 and trunc < 0.5:
            diff = 2  # Hard
        else:
            diff = -1  # Unknown/DontCare

        difficulty.append(diff)

    return {
        "name": np.array(names),
        "truncated": np.array(truncated, dtype=np.float32),
        "occluded": np.array(occluded, dtype=np.int32),
        "alpha": np.array(alpha, dtype=np.float32),
        "bbox": np.array(bbox, dtype=np.float32).reshape(-1, 4),
        "dimensions": np.array(dimensions, dtype=np.float32).reshape(-1, 3),
        "location": np.array(location, dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.array(rotation_y, dtype=np.float32),
        "score": np.array(score, dtype=np.float32),
        "difficulty": np.array(difficulty, dtype=np.int32),
    }


def camera_to_lidar(
    points: np.ndarray, r_rect: np.ndarray, velo2cam: np.ndarray
) -> np.ndarray:
    """Convert points from camera coordinate to LiDAR coordinate.

    Args:
        points: Points in camera coordinate (N, 3).
        r_rect: Rectification matrix (4, 4).
        velo2cam: Velodyne to camera transform (4, 4).

    Returns:
        Points in LiDAR coordinate (N, 3).
    """
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    lidar2cam = r_rect @ velo2cam
    cam2lidar = np.linalg.inv(lidar2cam)
    points_lidar = points_hom @ cam2lidar.T
    return points_lidar[:, :3]


def get_lidar_boxes(
    annos: Dict[str, np.ndarray], calib: Dict[str, np.ndarray]
) -> np.ndarray:
    """Convert camera 3D boxes to LiDAR coordinate.

    Args:
        annos: Annotation dict with location, dimensions, rotation_y.
        calib: Calibration dict.

    Returns:
        3D boxes in LiDAR coordinate (N, 7) [x, y, z, w, l, h, rot].
    """
    if len(annos["name"]) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    loc = annos["location"]  # (N, 3) in camera coord
    dims = annos["dimensions"]  # (N, 3) [h, w, l]
    rots = annos["rotation_y"]  # (N,)

    # Convert center from camera to LiDAR
    # Camera: x-right, y-down, z-forward
    # LiDAR: x-forward, y-left, z-up
    loc_lidar = camera_to_lidar(loc, calib["R0_rect"], calib["Tr_velo_to_cam"])

    # Adjust center height (camera gives bottom center, we want center)
    loc_lidar[:, 2] += dims[:, 0] / 2

    # Convert dimensions [h, w, l] to [w, l, h]
    # Note: in LiDAR coord, w=width(y), l=length(x), h=height(z)
    dims_lidar = dims[:, [1, 2, 0]]  # [w, l, h]

    # Convert rotation: camera rotation_y to LiDAR rotation
    # Camera: rotation around y-axis (down)
    # LiDAR: rotation around z-axis (up)
    rots_lidar = -rots - np.pi / 2

    boxes_lidar = np.concatenate(
        [loc_lidar, dims_lidar, rots_lidar[:, np.newaxis]], axis=1
    )

    return boxes_lidar.astype(np.float32)


def get_points_info(velodyne_path: str) -> Dict:
    """Get point cloud info without loading full data.

    Args:
        velodyne_path: Path to velodyne bin file.

    Returns:
        Dict with point cloud metadata.
    """
    if not osp.exists(velodyne_path):
        return {"num_pts": 0}

    # Get file size to estimate number of points
    file_size = os.path.getsize(velodyne_path)
    num_pts = file_size // (4 * 4)  # 4 floats per point, 4 bytes per float

    return {"num_pts": num_pts}


def process_single_sample(
    data_root: str,
    split: str,
    idx: str,
    with_label: bool = True,
) -> Optional[Dict]:
    """Process a single KITTI sample.

    Args:
        data_root: Root path of KITTI dataset.
        split: 'training' or 'testing'.
        idx: Sample index string (e.g., '000001').
        with_label: Whether to include labels.

    Returns:
        Info dict for this sample, or None if failed.
    """
    info = {}

    # Sample index
    info["sample_idx"] = int(idx)

    # Point cloud path
    velodyne_path = osp.join(data_root, split, "velodyne", f"{idx}.bin")
    if not osp.exists(velodyne_path):
        return None

    info["lidar_points"] = {
        "lidar_path": osp.join(split, "velodyne", f"{idx}.bin"),
        "num_pts_feats": 4,
    }

    # Get point count
    pts_info = get_points_info(velodyne_path)
    info["lidar_points"]["num_pts"] = pts_info["num_pts"]

    # Image path
    image_path = osp.join(data_root, split, "image_2", f"{idx}.png")
    if osp.exists(image_path):
        info["images"] = {
            "CAM2": {"img_path": osp.join(split, "image_2", f"{idx}.png")}
        }

    # Calibration
    calib_path = osp.join(data_root, split, "calib", f"{idx}.txt")
    if osp.exists(calib_path):
        calib = read_calib(calib_path)
        info["calib"] = {
            "P2": calib["P2"].tolist(),
            "R0_rect": calib["R0_rect"].tolist(),
            "Tr_velo_to_cam": calib["Tr_velo_to_cam"].tolist(),
        }
    else:
        return None

    # Labels (training only)
    if with_label and split == "training":
        label_path = osp.join(data_root, split, "label_2", f"{idx}.txt")
        annos = read_label(label_path)

        # Filter DontCare
        valid_mask = annos["name"] != "DontCare"
        for key in annos:
            annos[key] = annos[key][valid_mask] if len(annos[key]) > 0 else annos[key]

        # Convert to LiDAR boxes
        boxes_lidar = get_lidar_boxes(annos, read_calib(calib_path))

        info["annos"] = {
            "name": annos["name"].tolist(),
            "truncated": annos["truncated"].tolist(),
            "occluded": annos["occluded"].tolist(),
            "alpha": annos["alpha"].tolist(),
            "bbox": annos["bbox"].tolist(),
            "dimensions": annos["dimensions"].tolist(),
            "location": annos["location"].tolist(),
            "rotation_y": annos["rotation_y"].tolist(),
            "difficulty": annos["difficulty"].tolist(),
            "gt_bboxes_3d": boxes_lidar.tolist(),
        }

    return info


def create_kitti_info_file(
    data_root: str,
    split: str,
    sample_ids: List[str],
    out_path: str,
    with_label: bool = True,
):
    """Create KITTI info file for a split.

    Args:
        data_root: Root path of KITTI dataset.
        split: 'training' or 'testing'.
        sample_ids: List of sample index strings.
        out_path: Output pickle file path.
        with_label: Whether to include labels.
    """
    print(f"Processing {len(sample_ids)} samples for {split}...")

    infos = []
    for idx in tqdm(sample_ids, desc=f"Creating {osp.basename(out_path)}"):
        info = process_single_sample(data_root, split, idx, with_label)
        if info is not None:
            infos.append(info)

    print(f"  Valid samples: {len(infos)}/{len(sample_ids)}")

    # Save to pickle
    data = {
        "metainfo": {
            "dataset": "kitti",
            "version": "1.0",
            "classes": ["Car", "Pedestrian", "Cyclist"],
        },
        "data_list": infos,
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  Saved to {out_path}")


def main():
    args = parse_args()

    data_root = args.data_root
    out_dir = args.out_dir or data_root

    print("=" * 60)
    print("KITTI Data Converter")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")
    print()

    # Check data exists
    if not osp.exists(osp.join(data_root, "training", "velodyne")):
        print("ERROR: Velodyne data not found!")
        print(f"Please download KITTI data to {data_root}")
        print("See: tools/data_converter/download_kitti.sh")
        return

    # Read split files
    imagesets_dir = osp.join(data_root, "ImageSets")

    splits = {}
    for split_name in ["train", "val", "trainval", "test"]:
        split_file = osp.join(imagesets_dir, f"{split_name}.txt")
        if osp.exists(split_file):
            with open(split_file, "r") as f:
                splits[split_name] = [line.strip() for line in f.readlines()]
            print(f"Found {split_name}: {len(splits[split_name])} samples")
        else:
            print(f"Warning: {split_file} not found, skipping {split_name}")

    print()

    # Create info files
    os.makedirs(out_dir, exist_ok=True)

    if "train" in splits:
        create_kitti_info_file(
            data_root,
            "training",
            splits["train"],
            osp.join(out_dir, "kitti_infos_train.pkl"),
            with_label=True,
        )

    if "val" in splits:
        create_kitti_info_file(
            data_root,
            "training",
            splits["val"],
            osp.join(out_dir, "kitti_infos_val.pkl"),
            with_label=True,
        )

    if "trainval" in splits:
        create_kitti_info_file(
            data_root,
            "training",
            splits["trainval"],
            osp.join(out_dir, "kitti_infos_trainval.pkl"),
            with_label=True,
        )

    if "test" in splits:
        create_kitti_info_file(
            data_root,
            "testing",
            splits["test"],
            osp.join(out_dir, "kitti_infos_test.pkl"),
            with_label=False,
        )

    print()
    print("=" * 60)
    print("Done! Generated files:")
    for f in os.listdir(out_dir):
        if f.endswith(".pkl"):
            fpath = osp.join(out_dir, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"  {f}: {size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
