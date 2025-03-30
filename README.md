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
In addition, we will explorethe following research questions to enhance the project:
- How do different forms of weak supervision impact segmentation performance?
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
│   ├── config/             # Config folder to run the experiments
│   ├── models/             # Trained models
│   ├── cam_utils.py        # Class Activation Maps functions
│   ├── dataset.py          # Data loading and preprocessing for the Oxford-IIIT Pet dataset
│   ├── fit.py              # Training loop for optimizing model parameters
│   ├── metrics.py          # Functions for evaluating segmentation performance (Dice score, pixel accuracy, ...)
│   ├── models.py           # Various network architectures (UNet, DeepLabV3, FCN, PetClassifier, ...)
│   ├── utilities.py        # Helper functions for device management and resource cleanup
│   ├── visualization.py    # Visualizing predictions and comparing them to ground truth
├── .gitignore
├── baseline.py             # Baselines (fully supervised usecase)
├── main.py                 # Weak supervision segmentation experiments
├── README.md               # This file
```

## Getting started
### 1️. Environment setup
Create a conda env and install dependencies:
```sh
conda create --name weakseg python=3.9 -y && \
conda activate weakseg && \
conda install -y numpy matplotlib && \
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Baselines (fully supervised usecase)
Run the baseline script with the desired config (stored in a .json file in ``src\config``` **Modify the config to change the parameters**)
```sh
python python baseline.py --config .\src\config\baseline.json
```

### 3. Weak supervision segmentation
Run the main script with the desired config (stored in a .json file in ``src\config``` **Modify the config to change the parameters**)
```sh
python python main.py --config .\src\config\main.json
```


## References & resources
- **Project information**: see on Moodle the .pdf file
- **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Contributors
- Paul Hellegouarch
- Jules Talbourdet (primary contributor)