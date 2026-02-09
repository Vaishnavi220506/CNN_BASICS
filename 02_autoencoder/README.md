# Simple Autoencoder with PyTorch

This folder contains a beginner-friendly Jupyter Notebook that introduces the **fundamentals of Autoencoders** using **PyTorch**.  
The notebook focuses on understanding how autoencoders learn compressed representations of data and reconstruct inputs.

---

## Run in Google Colab

Click the button below to open the notebook directly in Google Colab:

[Open in Colab](https://colab.research.google.com/github/Vaishnavi220506/CNN_BASICS/blob/main/02_autoencoder/simple_autoencoder.ipynb)

---

## Topics Covered

* What is an Autoencoder
* Encoder and Decoder architecture
* Latent space representation
* Dimensionality reduction
* Image reconstruction
* Loss function for autoencoders
* Training an autoencoder using PyTorch

---

## Notebook

* `simple_autoencoder.ipynb`

---

## Notebook Structure with Key Syntax

### Step 1: Importing Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
Step 2: Dataset Preparation
Loads and preprocesses image data for training.

from torchvision import datasets, transforms
Step 3: Defining the Autoencoder Model
Creates encoder and decoder networks.

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
Step 4: Encoder
Compresses the input into a latent representation.

self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU()
)
Step 5: Decoder
Reconstructs the input from the latent vector.

self.decoder = nn.Sequential(
    nn.Linear(hidden_dim, input_dim),
    nn.Sigmoid()
)
Step 6: Forward Pass
Defines how data flows through the autoencoder.

def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
Step 7: Loss Function and Optimizer
Uses Mean Squared Error loss and Adam optimizer.

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
Step 8: Training Loop
Trains the autoencoder on input data.

loss.backward()
optimizer.step()
How to Run
Open the notebook using Google Colab, Jupyter Notebook, or VS Code

Run the cells from top to bottom

Observe how the model reconstructs the input images

Requirements
Python 3.x

PyTorch

torchvision

Install dependencies:

pip install torch torchvision
License
This project is licensed under the MIT License.
