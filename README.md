Advanced Deepfake Detection Framework

> **Dual-Stream Multi-Patch Ensemble Network (DSMPE-Net)**
> A robust forensic tool designed to detect generated media by analyzing both visual artifacts and frequency/spectral anomalies.

---

## 📚 Table of Contents
1.  **[The Concept](#-stage-1-the-concept)** - Why this exists and how it works.
2.  **[Architecture](#-stage-2-architecture)** - The engine under the hood.
3.  **[Installation](#-stage-3-installation)** - Setting up your environment.
4.  **[Training Strategy](#-stage-4-training-recommended)** - How to train on Kaggle (Free).
5.  **[Inference & Usage](#-stage-5-inference--usage)** - Running the model on videos/images.
6.  **[Datasets](#-datasets)** - What data we use.

---

## 🧠 Stage 1: The Concept

Deepfakes are becoming perfect. Traditional detectors that look for "blurry faces" or "weird eyes" are failing because modern GANs (Generative Adversarial Networks) are too good.

This model takes a different approach. It assumes that while a deepfake might *look* perfect, the mathematical process of generating it leaves behind invisible "fingerprints" in the **Frequency Domain** (DCT coefficients).

*   **Spatial Stream:** Looks at the image like a human (RGB pixels).
*   **Frequency Stream:** Looks at the image like a signal engineer (DCT transform).
*   **The Verdict:** If *either* stream detects an anomaly, the image is flagged.

---

## 🏗️ Stage 2: Architecture

We don't just look at the whole face. We break it down.

### 1. Patch Extraction
The face is cropped and split into **9 Overlapping Patches**. This forces the model to look at local details (texture of skin, edge of glasses, hair blending) rather than just the global structure.

### 2. Dual-Stream Network
Each patch goes through two parallel networks:
*   **Xception (RGB)**: Extracts visual features.
*   **ResNet50 (DCT)**: Extracts frequency artifacts.

### 3. Ensemble Decision
The system aggregates the scores from all 9 patches. If even **one** patch is highly suspicious (e.g., a glitch in the ear rendering), the whole video can be flagged.

---

## 🛠️ Stage 3: Installation

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU (Recommended for Inference)

### Quick Start
```bash
# 1. Clone the Repo
git clone https://github.com/MohithTP/DeepFake.git
cd DeepFake

# 2. Install Dependencies
# We recommend using 'uv' for speed, or partial installs.
pip install -r requirements.txt
```

---

## 🚂 Stage 4: Training (Recommended)

Since training Deep Learning models requires massive GPU power, we have optimized this project for **Kaggle Notebooks (Free Tesla T4 GPUs)**.


**How to Train:**
1.  **Open Kaggle:** Create a new Notebook.
2.  **Add Dataset:** Search for and add the `wild-deepfake` dataset.
3.  **Use the Script:** found in `kaggle_script.py`.
4.  **Run:** It will automatically:
    *   Clone this repo.
    *   Install dependencies.
    *   Train for 10 Epochs.
    *   Save `dsmpe_net_epoch_X.pth` after every epoch.

**Resume Training:**
If the notebook disconnects, simply download the last `.pth` file, upload it to a new notebook, and update the `resume_path` variable in the script.

---

## 🎬 Stage 5: Inference & Usage

Once you have your trained model (`dsmpe_net_final.pth`), you can run it on your local machine.

### 1. Analyzing a Video (Best for Demo)
This pipeline splits the video into frames, detects faces, runs the model, and gives a final Real/Fake score.

```bash
python -m src.video.parallel_processor \
  --video "path/to/suspect_video.mp4" \
  --model "dsmpe_net_final.pth" \
  --visualize
```
*   **Output:** It will print the score and save a `debug_patches.png` showing what it saw.

### 2. Analyzing a Single Image
```bash
python inference.py \
  --image_path "path/to/image.jpg" \
  --weight_path "dsmpe_net_final.pth"
```

---

## 📂 Datasets

*   **WildDeepfake:** Used for Training. Contains distinct real-world deepfakes from the internet.
*   **Celeb-DF v2:** Used for Testing. High-quality deepfakes that are hard to detect.

---

## 📜 License
MIT License. Free to use for research and educational purposes.
