# CNN Introduction (PyTorch)

This folder contains a beginner-friendly Jupyter Notebook that introduces the **basics of Convolutional Neural Networks (CNNs)** using **PyTorch**.  
It is intended for students starting with deep learning and computer vision.

---

## Run in Google Colab

Click the button below to open the notebook directly in Google Colab:

[Open in Colab](https://colab.research.google.com/github/Vaishnavi220506/CNN_BASICS/blob/main/01_CNN_INTRO/CNN_basics.ipynb)

---

## Topics Covered

* What is a Convolutional Neural Network (CNN)
* Why CNNs are used for image data
* Image tensors and dimensions
* Convolution layers
* Pooling layers
* Activation functions (ReLU)
* Fully connected layers
* Basic CNN architecture in PyTorch

---

## Notebook

* `CNN_basics.ipynb`

---

## Notebook Structure with Key Syntax

### Step 1: Importing Required Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
Step 2: Image Representation as Tensors
Images represented as tensors in the form:

(batch_size, channels, height, width)
Step 3: Defining a CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
Step 4: Convolution Layer
nn.Conv2d(in_channels, out_channels, kernel_size)
Step 5: Pooling Layer
nn.MaxPool2d(kernel_size, stride)
Step 6: Activation Function
F.relu(x)
Step 7: Fully Connected Layer
nn.Linear(in_features, out_features)
Step 8: Forward Pass
def forward(self, x):
    x = self.conv1(x)
    return x
How to Run
Open the notebook using Google Colab or Jupyter Notebook

Run each cell sequentially

Follow the explanations alongside the code

Requirements
Python 3.x

PyTorch

Install PyTorch:

pip install torch
License
This project is licensed under the MIT License.
