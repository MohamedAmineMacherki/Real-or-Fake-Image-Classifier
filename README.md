# DeepFake-Classifier

This project is an application designed to classify images as either "AI-generated" or "Real" using a custom deep learning model called **FaceNeSt**. The project is split into two main components:

1. **Backend**: Built with FastAPI to serve the trained model and handle image classification requests.
2. **Frontend**: A user-friendly interface built with Streamlit to upload images and display classification results.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
  - [Backend](#backend-setup)
  - [Frontend](#frontend-setup)
- [Model Architecture](#model-architecture)

---

## Features

- **Custom Model**: Implements a hybrid attention-based architecture called **FaceNeSt**.
- **GPU Support**: Utilizes CUDA (if available) for efficient processing.
- **FastAPI Backend**: Processes classification requests via REST APIs.
- **Streamlit Frontend**: Provides an interactive and intuitive user interface.

---

## Technologies Used

- **Python**: Main programming language.
- **FastAPI**: For backend API development.
- **Streamlit**: For frontend UI.
- **PyTorch**: For deep learning model implementation.
- **TorchVision**: For image processing.
- **Pillow**: For image handling.

---

## Directory Structure

```plaintext
project/
│
├── api/
│   ├── faceNest.py          # Model definition and utilities
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Backend dependencies
│
├── frontend/
│   ├── app.py               # Streamlit frontend
│   ├── requirements.txt     # Frontend dependencies
│
└── model/
    └── best_model.pth       # Pre-trained model weights

```

## Setup Instructions

### Backend Setup

1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Classify Images:

- Upload an image via the Streamlit interface.
- The frontend sends the image to the backend for processing.
- The classification result ("AI-generated" or "Real") is displayed on the interface.

---

## Model Architecture

The **FaceNeSt** model integrates advanced attention mechanisms to improve classification performance.

### Key Components:

- **GLCSAttention**: Captures both local and global channel-spatial attention.
- **AdaptivelyWeightedMultiScaleAttention**: Fuses multi-scale features with dynamically learned weights.
- **ResNet-inspired Blocks**: Provides hierarchical feature extraction and representation learning.

This design ensures robust discrimination between real and AI-generated images.

---
