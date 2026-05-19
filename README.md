# Custom PyTorch CLIP Implementation

A complete, from-scratch PyTorch implementation of OpenAI's **CLIP (Contrastive Language-Image Pretraining)** model. This project features a custom Vision Transformer (ViT) for image encoding, a Transformer-based text encoder, and a highly efficient streaming data pipeline designed to handle large-scale multimodal datasets without memory bottlenecks.

## 🚀 Project Overview

This model maps images and text into a shared 512-dimensional latent space using a symmetric contrastive loss function. It is built entirely from the ground up to demonstrate a deep understanding of Vision-Language Model (VLM) architectures.

* **Total Parameters:** ~210M (209,979,136)
* **Dataset:** Conceptual Captions 3M (`WeiChow/cc3m`) via Hugging Face
* **Tokenizer:** `openai/clip-vit-base-patch32`
* **Framework:** PyTorch

## 📊 Dataset & Streaming Pipeline

To handle large-scale data efficiently without requiring massive local storage, this project utilizes the Hugging Face `datasets` library in **streaming mode**. 

The dataset is cleanly split to ensure zero data leakage:
* **Training Set:** 100,000 image-text pairs (`dataset['train'].take(100000)`)
* **Validation Set:** 5,000 image-text pairs (`dataset['train'].skip(100000).take(5000)`)

The custom `IterableDataset` (`CLIPDataset`) fetches images on the fly, applies `torchvision` augmentations (resizing, horizontal flips, normalization), and tokenizes the text seamlessly during the training loop.

## 🧠 Model Architecture

The model consists of two primary encoders and projection heads:

### 1. Image Encoder (Vision Transformer)
* **Input Image Size:** 224x224
* **Patch Size:** 16x16
* **Layers:** 12 | **Attention Heads:** 8
* **Hidden Dimension:** 768 (MLP hidden dim: 3072)

### 2. Text Encoder (Transformer)
* **Max Sequence Length:** 512
* **Vocabulary Size:** 49,408
* **Layers:** 12 | **Attention Heads:** 8
* **Hidden Dimension:** 768 (MLP hidden dim: 3072)

### 3. Projection Layers
Both the ViT and the Text Transformer output a 768-dimensional vector. These are passed through a linear projection layer with dropout (0.1) to reach the final **512-dimensional** shared embedding space.

## 📂 Repository Structure

The codebase is highly modular for readability and easy debugging:

* `config.ipynb`: Hyperparameter dictionaries for image, text, and projection modules.
* `image_encoder.ipynb`: ViT components (`PatchEmbedding`, `MultiHeadAttention`, `MLP`, `Normalization`).
* `text_encoder.ipynb`: Text components (`Embedding`, `PositionalEncoding`, `FeedForward`, `MultiHeadAttention`).
* `clip_model.ipynb`: The main `CLIP` class bridging the encoders and projection layers.
* `contrastive_loss.ipynb`: Symmetric cross-entropy loss with a learnable temperature parameter.
* `clip_dataset.ipynb`: PyTorch `IterableDataset` handling the streaming data and augmentations.
* `data_pipeline.ipynb`: Environment setup, Hugging Face authentication, and dataset loading.
* `training.ipynb`: The main training loop, optimizer setup, and checkpointing logic.
* `inference.ipynb`: Script for loading weights, calculating cosine similarity, and running zero-shot image-text matching.

## ⚙️ Training Details

* **Optimizer:** `AdamW` (Learning rate: `1e-4`, Weight decay: `0.01`)
* **Scheduler:** `CosineAnnealingLR` (T_max: 60 epochs)
* **Loss Function:** Custom Symmetric Contrastive Loss (initial temperature: 0.07)
* **Batch Size:** 32
* **Checkpointing:** Automatically saves the best model based on validation loss, with early stopping triggered after 5 epochs of no improvement.

## 💻 Usage & Inference

The complete zero-shot inference pipeline is provided in **`inference.ipynb`**. 

This notebook demonstrates how to:
1. Load the trained model weights.
2. Tokenize text captions using the Hugging Face tokenizer.
3. Preprocess and transform input images.
4. Extract features from both encoders and calculate the cosine similarity matrix for zero-shot image-text matching.
