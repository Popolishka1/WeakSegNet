# WeakSegNet: Weakly-supervised neural networks for image segmentation

## Project overview
This repository contains the implementation of **WeakSegNet**, a weakly-supervised deep learning framework for image segmentation. Developed as part of **COMP0197: Applied Deep Learning** group project.

Our implementation is evaluated on the **Oxford-IIIT Pet Dataset**.

## Research scope
### General goal
- Define a weak supervision problem
- Select a weakly-supervised segmentation algorithm
- Implement a weakly-supervised segmentation network
- Compare performance with a **fully-supervised method** and conduct an **ablation study** on key hyperparameters

### Extension
In addition, we will explore the following research questions to enhance the project:
- What are the key limitations of pseudo mask generation in WSSS, and how can foundation models help overcome them?
- Conduct controlled experiments to evaluate the segmentation quality under weak labels

## Dataset
- **Oxford-IIIT Pet Dataset** ([Link](https://www.robots.ox.ac.uk/~vgg/data/pets/))
- Contains thousands of images of 37 pet breeds with pixel-wise segmentation labels

## Repository structure
```
├── data/                   # Raw dataset files, including annotations and images
│   ├── annotations   
│   ├── images
├── notebooks/              # Jupyter notebooks for running experiments and quick prototyping
├── results/                # Stores outputs and visualizations from experiments
├── src/                    # Core codebase for WeakSegNet
│   ├── configs/            # Configs folder to run the experiments
│   ├── models/             # Trained models
│   ├── bbox_utils.py       # Generate weak mask from bounding boxes
│   ├── cam_utils.py        # Class Activation Maps functions
│   ├── classification.py   # Classification network architectures
│   ├── dataset.py          # Data loading and preprocessing (transformations/augmentations) for the Oxford-IIIT Pet dataset
│   ├── fit.py              # Training loop for optimizing model parameters (classifier, segmentation model, ...)
│   ├── fm_utils.py         # Generate weak mask from foundation models
│   ├── metrics.py          # Functions for evaluating segmentation model performance and classifier accuracy
│   ├── models.py           # Various network architectures (UNet, DeepLabV3, FCN, RedNet, ...)
│   ├── utils.py            # Helper functions for device management and resource cleanup
│   ├── visualization.py    # Visualization tools
├── .gitignore
├── baseline.py             # Baselines (fully supervised use case)
├── main_bbox.py            # Weak supervision segmentation (bbox) experiments
├── main_cam.py             # Weak supervision segmentation (CAM) experiments
├── main_fm.py              # Weak supervision segmentation (FM) experiments
├── README.md               # This file
```

## Getting started and running the code

### 1️. Environment setup
Using the comp0197-cw1-pt conda environment, install the following two package: 
```sh
pip install opencv-python
```
```sh
pip install sam2
```

### 2. Baselines (fully supervised use case)
Run the baseline script with the desired config (stored in a .json file in ```src/config``` **Modify the config to change the parameters**)
```sh
python python baseline.py --config ./src/config/baseline.json
```

### 3. Weak supervision segmentation : Class Activation Map
Run the main script with the desired config (stored in a .json file in ```src/configs``` **Modify the config to change the parameters**)
```sh
python python main_cam.py --config ./src/configs/main_cam.json
```

### 4. Weak supervision segmentation : Bounding Box
Run the main script with the desired config (stored in a .json file in ```src/configs``` **Modify the config to change the parameters**)
```sh
python python main_bbox.py --config ./src/configs/main_bbox.json
```

### 4. Weak supervision segmentation : Foundation Models
Run the main script with the desired config (stored in a .json file in ```src/configs``` **Modify the config to change the parameters**)
```sh
python python main_fm.py --config ./src/configs/main_fm.json
```

Zero-shot experiment: 
```sh
python main_fm.py --config ./src/configs/main_fm.json
```
with the last line of main_fm.py: main(mode = "zero_shot").

Pseudo-mask experiment:
```sh
python main_fm.py --config ./src/configs/main_fm.json
```
with the last line of main_fm.py: main(mode = "pseudo_mask").

CAM-SAM experiment:
```sh
python main_fm.py --config ./src/configs/cam_sam_config.json
```
with the last line of main_fm.py: main(mode = "cam_pseudo_mask").


## References & resources
- **Project information**: see on Moodle the .pdf file
- **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
