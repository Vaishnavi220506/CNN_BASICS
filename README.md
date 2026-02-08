# CNN Basics with PyTorch

This repository contains a beginner-friendly Jupyter Notebook that introduces the **fundamentals of Convolutional Neural Networks (CNNs)** using **PyTorch**.  
It is designed for students who are starting with deep learning and computer vision.

---

## Run in Google Colab

Click the button below to open the notebook directly in Google Colab:

[Open in Colab](https://colab.research.google.com/)

> You can upload the notebook manually to Colab or replace the link with your GitHub notebook link.

---

## Topics Covered

* Introduction to Convolutional Neural Networks
* Why CNNs are used for images
* Tensors and image data representation
* Convolution layers
* Pooling layers
* Activation functions (ReLU)
* Fully connected layers
* Forward pass of a CNN
* Basic CNN architecture in PyTorch

---

## Notebook

* `CNN_basics.ipynb`

---

## Notebook Structure with Key Syntax

### Step 1: Importing Required Libraries

Imports PyTorch and supporting libraries.

**Syntax:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
````

---

### Step 2: Understanding Image Tensors

Explains how images are represented as tensors
`(channels, height, width)`.

---

### Step 3: Defining a CNN Model

Creates a CNN using PyTorch’s `nn.Module`.

**Syntax:**

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
```

---

### Step 4: Convolution Layer

Applies convolution to extract features from images.

**Syntax:**

```python
nn.Conv2d(in_channels, out_channels, kernel_size)
```

---

### Step 5: Pooling Layer

Reduces spatial dimensions of feature maps.

**Syntax:**

```python
nn.MaxPool2d(kernel_size, stride)
```

---

### Step 6: Activation Function (ReLU)

Introduces non-linearity into the model.

**Syntax:**

```python
F.relu(x)
```

---

### Step 7: Fully Connected Layers

Maps extracted features to output classes.

**Syntax:**

```python
nn.Linear(in_features, out_features)
```

---

### Step 8: Forward Function

Defines the forward pass of the CNN.

**Syntax:**

```python
def forward(self, x):
    x = self.conv1(x)
    return x
```

---

### Step 9: Model Summary

Creates an instance of the CNN model.

**Syntax:**

```python
model = CNN()
print(model)
```

---

## How to Run

1. Open the notebook in **Google Colab**, **Jupyter Notebook**, or **VS Code**
2. Run the cells step by step from top to bottom
3. Read the explanations alongside the code

---

## Requirements

* Python 3.x
* PyTorch

Install PyTorch using pip:

```bash
pip install torch
```

---

## License

This project is licensed under the **MIT License**.

```

---

If you want, I can:
- 🔗 Replace the Colab link with your **actual GitHub link**
- ✍️ Customize it **exactly** for CIFAR-10 / Cats vs Dogs / any dataset
- 🧠 Make a **next-level README** (with diagrams + learning outcomes)

Just tell me 😄
```
