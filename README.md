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
├── data/
│   ├── annotations   
│   ├── images
├── notebooks/          # Jupyter notebooks for experimentation
├── results/
├── src/                # Codebase for WeakSegNet implementation
│   ├── __init__.py 
│   ├── baseline.py 
│   ├── dataset.py 
│   ├── fit.py 
│   ├── main.py 
│   ├── metrics.py 
│   ├── models.py 
│   ├── utilities.py 
│   ├── visualization.py 
├── .gitignore
├── README.md           # This file
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

### 2. Training the model
Run the training script (only a **baseline** UNet for now. Set ``Train`` to True to train)
```sh
python src/main.py
```

### 3. Evaluating performance
Evaluate model predictions and compare against baselines: TBD


## References & resources
- **Project information**: see on Moodle the .pdf file
- **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Contributors
- Paul Hellegouarch
- Jules Talbourdet (primary contributor)