# Data Augmentation with PyTorch

This folder contains a beginner-friendly Jupyter Notebook that explains image data augmentation techniques using PyTorch and Torchvision.  
The notebook demonstrates how data augmentation increases dataset diversity and improves the generalization ability of deep learning models.

---

## Run in Google Colab

Click the button below to open the notebook directly in Google Colab:

[Open in Colab](https://colab.research.google.com/github/Vaishnavi220506/CNN_BASICS/blob/main/03_data_augmentation/dataaug.ipynb)

---

## Topics Covered

- What is Data Augmentation  
- Why Data Augmentation is important  
- Image preprocessing basics  
- Using `torchvision.transforms`  
- Common augmentation techniques:
  - Random Horizontal Flip  
  - Random Rotation  
  - Random Crop  
  - Normalization  
  - Convert images to tensors  
- Visualizing original vs augmented images  

---

## Notebook

- `dataaug.ipynb`

---

## Notebook Structure with Key Syntax

### Step 1: Importing Required Libraries
```python
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
Step 2: Defining Transformations
Applies a sequence of augmentation techniques.

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
Step 3: Loading the Dataset
Loads image data with applied augmentations.

dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
Step 4: DataLoader
Creates batches for training.

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
Step 5: Visualizing Augmented Images
Displays how the same image changes after augmentation.

images, labels = next(iter(loader))
How to Run
Open the notebook using Google Colab, Jupyter Notebook, or VS Code

Run the cells sequentially from top to bottom

Observe the effect of each augmentation visually

Requirements
Python 3.x

PyTorch

torchvision

matplotlib

numpy

Install dependencies:

pip install torch torchvision matplotlib numpy
License
This project is licensed under the MIT License.
