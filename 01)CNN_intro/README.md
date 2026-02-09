# Autoencoder using PyTorch

This project is a simple implementation of an autoencoder using PyTorch.  
It is trained on the MNIST dataset to learn how to compress images and reconstruct them.

## Project Overview

- Uses the MNIST handwritten digit dataset  
- Encoder reduces image features into a latent vector  
- Decoder reconstructs the image from the latent vector  
- Implemented using PyTorch  

## Model Architecture

- Input: 28 x 28 grayscale images  
- Encoder: Fully connected layers  
- Latent space: Compressed representation of the image  
- Decoder: Reconstructs the image back to original size  

## Dataset

- MNIST dataset  
- 60,000 training images  
- 10,000 test images  
- Image size: 28 x 28  

## Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repository-link>

## Open in Google Colab

Click the button below to open the notebook directly in Google Colab:

[Open in Colab](https://colab.research.google.com/github/Vaishnavi220506/CNN_BASICS/blob/main/01%29CNN_intro/simple_autoencoder.ipynb)

Output
Reconstructed MNIST images

Comparison between original and reconstructed images

Training loss decreases over epochs

Purpose
This project is made for learning autoencoders and understanding:

Feature extraction

Dimensionality reduction

Image reconstruction
