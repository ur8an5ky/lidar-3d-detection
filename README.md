# LiDAR 3D Object Detection

University project for the **Machine Learning** course — topic #2: object
detection in data from autonomous-vehicle drives, based on LiDAR point clouds.

## Team
This project is developed by two people:
- **Jakub Urbański** — data & pipeline
- **Artur Saganowski** — model & training

## Goal
Detect 3D objects (cars, pedestrians, cyclists) in LiDAR point clouds recorded
from a moving vehicle.

- **Input:** a single LiDAR scan — a point cloud with `(x, y, z, intensity)` per point.
- **Output:** a list of detected objects, each described by a 3D bounding box
  and a class label.

## Project structure
```
lidar-3d-detection
├── data/         # dataset (ignored by git — too large to commit)
├── notebooks/    # data exploration & visualizations
├── src/          # source code: data loading, preprocessing, utilities
├── configs/      # model / training configuration files
├── results/      # final plots & visualizations for the presentation
├── requirements.txt
└── README.md
```