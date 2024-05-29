<h1 align="center">PPE Detection with YOLOv8</h1>

## Overview

This repository provides the code for training a custom YOLOv8 model with custom data to detect whether individuals are wearing personal protective equipment (PPE) in videos. YOLOv8 (You Only Look Once) is a state-of-the-art, real-time object detection system.

## Examples

<p align="center">
  <img src="https://github.com/mrkrisgee/PPE_detection/blob/main/gifs/ppe_1_results.gif" alt="YOLOv8 PPE Detection 1">
</p>
<p align="center">
  <img src="https://github.com/mrkrisgee/PPE_detection/blob/main/gifs/ppe_2_results.gif" alt="YOLOv8 PPE Detection 2">
</p>
<p align="center">
  <img src="https://github.com/mrkrisgee/PPE_detection/blob/main/gifs/ppe_3_results.gif" alt="YOLOv8 PPE Detection 3">
</p>

## Usage

## 1. Training the Model

### Download Construction Site Safety Image Dataset

Download the dataset from the following link:

```
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30/download/yolov8
```

### Organize Dataset in Google Drive

Add the dataset to your Google Drive with the following hierarchy:

<img src="https://github.com/mrkrisgee/PPE_detection/blob/main/extras/gDrive.png" width="608">

### Train the Model

Download and run the training notebook:

```
https://github.com/mrkrisgee/PPE_detection/blob/main/train_construction_site_image_dataset.ipynb
```

### Retrieve and Download the Best Model

```
This will be saved in: `/runs/detect/train/weights/best.pt`
```

## 2. Making Predictions on Videos

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) installed on your system. Anaconda simplifies package management and deployment.

### Create a Virtual Environment

Create and activate a new conda environment by running the following commands in your terminal:

```
conda create -n yolov8
conda activate yolov8
```

### Clone the repository

Clone this repository to your local machine and navigate into the project directory:

```
git clone https://github.com/mrkrisgee/PPE_detection.git
cd PPE_detection
```

### Move the Trained Model

Move the `best.pt` model to the `/PPE_detection/model/ directory`.

### Install Necessary Packages

Install the required Python packages using pip:

```
pip install -r requirements.txt
```

### Download CUDA Toolkit

If you have an NVIDIA GPU and want to utilize CUDA for acceleration, download and install the CUDA toolkit from the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page.

```
https://developer.nvidia.com/cuda-downloads
```

### Run the Scripts

To execute the PPE detection script, run:

```
python PPE_detection.py
```

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): YOLOv8 is a real-time object detection model developed by Ultralytics.
- [Murtaza Hassan](https://github.com/murtazahassan): For his comprehensive Object Detection 101 course
