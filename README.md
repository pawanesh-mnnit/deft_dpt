# DEFT-DPT
**Dynamic Egocentric Feature Transformation (DEFT)** + **Dynamic Percentile Thresholding (DPT)**  
Research code for egocentric multimodal action recognition (RGB, Optical Flow, Depth) with graph-based learning (GAT / GCN).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Folder structure](#folder-structure)
- [Installation](#installation)

---

## Project Overview
This repository contains code to reproduce experiments for egocentric action recognition using:
- **DEFT** — a spatial-localization feature transformation module that adapts local receptive fields for egocentric frames.
- **DPT** — Dynamic Percentile Thresholding, a percentile-based graph sparsification method for constructing adjacency matrices from pairwise similarities.
- Graph Neural Networks (GAT / GCN) to model inter-frame and multimodal relationships.

The pipeline typical flow:
1. Input Frame
2. DEFT spatial transformer for preprocessing
3. Backbone feature extraction (e.g., ResNet50 / EfficientNet)
4. Sparse Video Similarity Graph (SVSG) construction using DPT
5. GNN modeling (GAT / GCN)
6. Training / evaluation / inference

This codebase is intended for research use (experimentation, ablations, reproducible results).

---

## Key Features
- DEFT implementation for spatially-adaptive feature transformation.
- Dynamic Percentile Thresholding (DPT) to create Sparse Video Similarity Graphs (SVSG) without choosing a fixed K.
- Support for multimodal inputs (RGB, Optical Flow, Depth).
- GAT and GCN model implementations.
- Training scripts with checkpointing, logging and resume capability.
- Evaluation scripts for Top-1, Top-5, Precision, Recall, F1-score, and confusion matrix.

---

## Requirements
- Python 3.8+ (3.9 recommended)
- PyTorch 1.10+ (install matching CUDA if available)
- PyTorch Geometric (matching your PyTorch & CUDA)
- Commonly used libs: numpy, scipy, scikit-learn, opencv-python, pyyaml, tqdm, pandas

## Folder Structure
- Input_Data
- Feature Extraction
- Model Training
- SavedModels
- Results

A `requirements.txt` is included. Install dependencies with:
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
# Install torch & torchvision separately per your CUDA: https://pytorch.org/
# Install PyG (PyTorch Geometric) per instructions: https://pytorch-geometric.readthedocs.io/
