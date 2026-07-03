# 🕵️‍♂️ Advanced Deepfake & Tamper Detection Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

A comprehensive, state-of-the-art forensic framework designed to detect AI-generated faces (deepfakes) and text document tampering. It combines an Intelligent Agentic Switchboard for dynamic routing with robust deep neural networks analyzing both visual and frequency-domain anomalies.

---

## 📖 Table of Contents
1. [Overview & Concept](#-overview--concept)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Installation & Prerequisites](#-installation--prerequisites)
5. [Usage & Inference](#-usage--inference)
6. [Training Strategy](#-training-strategy)
7. [Datasets](#-datasets)
8. [License](#-license)

---

## 🧠 Overview & Concept

As Generative AI models (like GANs and Diffusion models) reach photorealism, traditional detectors looking for visual blurriness are failing. 

This framework takes a multi-modal approach:
1. **Facial Deepfakes:** Assumes that generating deepfakes leaves behind invisible mathematical "fingerprints". We analyze images not just as RGB pixels, but as frequency signals using Discrete Cosine Transform (DCT) coefficients.
2. **Text Document Tampering:** Utilizes Error Level Analysis (ELA) to highlight compression artifacts introduced when text is digitally pasted or manipulated on a document.

---

## ✨ Key Features

* **Intelligent Routing:** An Agentic AI switchboard (powered by `agno`) analyzes incoming media and routes it to the optimal processing pipeline, saving computational power.
* **Dual-Stream Facial Analysis (DSMPE-Net):** Analyzes both Spatial (Xception) and Frequency (ResNet50) domains simultaneously.
* **Multi-Patch Extraction:** Splits faces into 9 overlapping patches to force the model to scrutinize microscopic, local details rather than holistic structures.
* **Text Tampering Detection:** A specialized Xception-based module focusing on ELA maps to catch forged documents.
* **Real-time Web UI:** A Flask and SocketIO powered frontend that provides users with live streaming logs of the backend forensic process.

---

## 🏗️ System Architecture

The pipeline consists of four major interconnected components:

### 1. The Agentic Router (Intelligent Switchboard)
Before media touches the heavy neural networks, it is triaged by a smart dispatcher:
- Evaluates metadata and uses basic CV tools to check image quality (blurriness) and scan for adversarial noise.
- Analyzes text density to identify if the image is a document.
- Routes the input optimally (e.g., `VIDEO_PIPELINE`, `FACE_PIPELINE`, `TEXT_TAMPER`, or `REJECT`).

### 2. The Facial Deepfake Engine (DSMPE-Net)
When a face or video is detected:
- **Deduplication & Extraction:** Uses dHash to skip identical video frames and MTCNN to accurately extract facial bounding boxes.
- **Dual-Stream Processing:** Each face is split into 9 patches. Every patch runs through an **Xception network** (for RGB visual artifacts) and a **ResNet50 network** (for DCT frequency anomalies).
- **Ensemble Verdict:** The network aggregates scores across all patches and streams. A single highly suspicious patch flags the entire media.

### 3. The Text Tamper Engine
When a document is detected (`TEXT_TAMPER` route):
- **Error Level Analysis (ELA):** The document undergoes ELA preprocessing to expose compression disparities.
- **Forensic Xception:** The ELA tensor is passed through a binary classification Xception network tailored to spot text manipulation.

### 4. The Web Controller
A **Flask / Flask-SocketIO** backend acts as the orchestrator, accepting uploads and pushing asynchronous, real-time forensic progress updates to the web interface.

---

## 🛠️ Installation & Prerequisites

### Prerequisites
*   Python 3.10 or higher
*   NVIDIA GPU with CUDA support (Highly Recommended for inference speed)
*   Required Weights Files:
    *   `dsmpe_net_final.pth` (Main Deepfake Model)
    *   `xception_ela_doctamper_latest.pth` (Text Tamper Model)

### Quick Start
```bash
# 1. Clone the Repository
git clone https://github.com/MohithTP/DeepFake.git
cd DeepFake

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Add Model Weights
# Ensure your .pth files are placed in the root directory.
```

---

## 🎬 Usage & Inference

### 1. Launching the Web Interface (Recommended)
The easiest way to interact with the full Agentic Router and detection pipelines.
```bash
python app.py
```
*Navigate to `http://localhost:5000` in your web browser. Upload an image or video and watch the real-time processing logs stream via WebSockets.*

### 2. CLI: Analyzing a Video
Splits a video into frames, detects faces, runs the model, and outputs a final score.
```bash
python -m src.video.parallel_processor \
  --video "path/to/suspect_video.mp4" \
  --model "dsmpe_net_final.pth" \
  --visualize
```

### 3. CLI: Analyzing a Single Image
```bash
python inference.py \
  --image_path "path/to/face_image.jpg" \
  --weight_path "dsmpe_net_final.pth"
```

---

## 🚂 Training Strategy

Due to the heavy computational requirements of deep learning, we recommend training on **Kaggle Notebooks (Free Tesla T4 GPUs)**.

1. **Setup:** Create a new Kaggle Notebook and attach the `wild-deepfake` dataset.
2. **Execute Script:** Run the provided `kaggle_script.py` within the notebook.
3. **Automated Pipeline:** The script will automatically clone this repository, install dependencies, and train for 10 epochs, saving checkpoints (`dsmpe_net_epoch_X.pth`) iteratively.
4. **Resuming:** If the kernel disconnects, download the latest checkpoint, re-upload it, and update the `resume_path` variable to continue training.

---

## 📂 Datasets

*   **WildDeepfake:** Used for Training. Contains distinct real-world deepfakes sourced from the internet.
*   **Celeb-DF v2:** Used for Testing/Validation. High-quality deepfakes that challenge traditional detectors.
*   **DocTamperV1:** (SCD/FCD) Used for training the Error Level Analysis text tampering model.

---

## 📜 License
This project is licensed under the **MIT License**. It is free to use, modify, and distribute for research and educational purposes. See the `LICENSE` file for more details.
