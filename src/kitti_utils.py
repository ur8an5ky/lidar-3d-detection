from pathlib import Path

import numpy as np
import pandas as pd

# Standard KITTI 3D object detection classes.
KITTI_CLASSES = [
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram", "Misc", "DontCare",
]

# Columns of a label_2/*.txt file, per the official KITTI devkit.
LABEL_COLUMNS = [
    "type", "truncated", "occluded", "alpha",
    "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",   # 2D box (pixels)
    "dim_height", "dim_width", "dim_length",                # 3D size (m)
    "loc_x", "loc_y", "loc_z",                              # location, camera coords (m)
    "rotation_y",                                           # yaw around camera Y (rad)
]


def load_point_cloud(bin_path):
    """Load a Velodyne .bin scan as an (N, 4) array: x, y, z, reflectance."""
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points


def load_labels(label_path):
    """Parse one label_2 .txt file into a DataFrame (one row per object)."""
    rows = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            obj = {"type": parts[0]}
            for col, val in zip(LABEL_COLUMNS[1:], parts[1:]):
                obj[col] = float(val)
            rows.append(obj)
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def load_calib(calib_path):
    """Parse a calibration .txt file into a dict of numpy matrices."""
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()])
    # Reshape the matrices you'll use most often.
    calib["P2"] = calib["P2"].reshape(3, 4)
    calib["R0_rect"] = calib["R0_rect"].reshape(3, 3)
    calib["Tr_velo_to_cam"] = calib["Tr_velo_to_cam"].reshape(3, 4)
    return calib


def collect_all_labels(label_dir):
    """Load every label file into one DataFrame, with a `frame_id` column.

    Use this for dataset-wide statistics: class distribution, object
    dimensions, distance histograms, etc.
    """
    label_dir = Path(label_dir)
    frames = []
    for txt in sorted(label_dir.glob("*.txt")):
        df = load_labels(txt)
        df["frame_id"] = txt.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Quick sanity check — adjust the path to your data.
    root = Path("data/kitti/training")
    pc = load_point_cloud(root / "velodyne" / "000000.bin")
    print(f"Point cloud shape: {pc.shape}  (expected: N x 4)")
    print(f"x range: [{pc[:, 0].min():.1f}, {pc[:, 0].max():.1f}] m")

    labels = collect_all_labels(root / "label_2")
    print(f"\nTotal objects: {len(labels)}")
    print("\nClass distribution:")
    print(labels["type"].value_counts())