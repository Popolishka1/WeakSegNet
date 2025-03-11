# WeakSegNet: Weakly-supervised neural networks for image segmentation

## Project overview
This repository contains the implementation of **WeakSegNet**, a weakly-supervised deep learning framework for image segmentation. Developed as part of **COMP0197: Applied Deep Learning** group project.

Our implementation is evaluated on the **Oxford-IIIT Pet Dataset**.

## Research scope
### General goal
- Define a weak supervision problem
- Select a weakly-supervised segmentation algorithm
- Implement a **weakly-supervised segmentation network**
- Compare performance with a **fully-supervised method** and conduct an **ablation study** on key hyperparameters

### Extension
In addition, we will explorethe following research questions to enhance the project:
- How do different forms of weak supervision impact segmentation performance?
- Conduct controlled experiments to evaluate the segmentation quality under weak labels.

## Dataset
- **Oxford-IIIT Pet Dataset** ([Link](https://www.robots.ox.ac.uk/~vgg/data/pets/))
- Contains thousands of images of 37 pet breeds with pixel-wise segmentation labels.

## Repository Structure
TBD
```
├── src/                # Codebase for WeakSegNet implementation
│   ├── fromage.py      # Deep learning model
├── notebooks/          # Jupyter notebooks for experimentation
├── results/
├── README.md           # This file
```

## Getting started
### 1️. Environment setup
Create a Conda environment and install dependencies:
```sh
conda create --name weakseg python=3.9 -y && \
conda activate weakseg && \
conda install -y numpy pandas matplotlib tqdm && \
conda install -y -c conda-forge opencv scikit-learn && \
conda install -y pytorch torchvision torchaudio -c pytorch
```

### 2. Training the model
Run the training script: TBD
```sh
python src/tbd.py
```

### 3. Evaluating performance
Evaluate model predictions and compare against baselines: TBD
```sh
python src/tbd.py
```

## References & Resources
- **Project information**: see on Moodle
- **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Contributors
- Name 1
- Name 2


