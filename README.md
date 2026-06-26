Advanced Deepfake Detection Framework

> **Dual-Stream Multi-Patch Ensemble Network (DSMPE-Net)**
> A robust forensic tool designed to detect generated media by analyzing both visual artifacts and frequency/spectral anomalies.

---

## 📚 Table of Contents
1.  **[The Concept](#-stage-1-the-concept)** - Why this exists and how it works.
2.  **[System Architecture](#-stage-2-system-architecture)** - The complete pipeline (Router, Engine, Web App).
3.  **[Installation](#-stage-3-installation)** - Setting up your environment.
4.  **[Training Strategy](#-stage-4-training-recommended)** - How to train on Kaggle (Free).
5.  **[Inference & Web Usage](#-stage-5-inference--web-usage)** - Running the web app and local scripts.
6.  **[Datasets](#-datasets)** - What data we use.

---

## 🧠 Stage 1: The Concept

Deepfakes are becoming perfect. Traditional detectors that look for "blurry faces" or "weird eyes" are failing because modern GANs (Generative Adversarial Networks) are too good.

This model takes a different approach. It assumes that while a deepfake might *look* perfect, the mathematical process of generating it leaves behind invisible "fingerprints" in the **Frequency Domain** (DCT coefficients).

*   **Spatial Stream:** Looks at the image like a human (RGB pixels).
*   **Frequency Stream:** Looks at the image like a signal engineer (DCT transform).
*   **The Verdict:** If *either* stream detects an anomaly, the image is flagged.

---

## 🏗️ Stage 2: System Architecture

The project consists of three main components working together to process media from upload to final verdict:

### 1. The Intelligent Switchboard (Agentic Router)
Before media reaches the heavy neural networks, it passes through an **Agentic Router** (built with `agno`). 
- Extracts metadata and uses computer vision tools to evaluate image quality (e.g., blurriness) and check for adversarial noise.
- Triages the input and optimally routes it (e.g., `VIDEO_PIPELINE`, `FACE_PIPELINE`, `TEXT_TAMPER`, or `REJECT`) to save computational resources.

### 2. The Model Handler Pipeline
When media is routed for deepfake analysis, the backend pipeline takes over:
- **Deduplication (dHash):** Skips duplicate frames in videos to optimize speed.
- **Face Extraction (MTCNN):** Isolates the face from the frame.

### 3. The Core AI Engine (DSMPE-Net)
The extracted face is passed to the **Dual-Stream Multi-Patch Ensemble Network**:
- **Patch Extraction:** The face is split into **9 Overlapping Patches** to scrutinize microscopic local details.
- **Dual-Stream Network:** Each patch is analyzed simultaneously by **Xception** (RGB/Visual artifacts) and **ResNet50** (DCT/Frequency anomalies).
- **Ensemble Decision:** Scores from all 9 patches and both streams are aggregated. If even one patch exhibits highly suspicious signals, the media is flagged.

### 4. The Web Application
A frontend built with **Flask** and **Flask-SocketIO** provides a user-friendly interface. It allows users to upload suspect media and utilizes WebSockets to stream real-time progression updates back to the browser as the Router and AI Engine process the file.

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

## 🎬 Stage 5: Inference & Web Usage

Once you have your trained model (`dsmpe_net_final.pth`), you can run the system.

### 1. Running the Web Application (Recommended)
Launch the Flask web application to interact with the full Agentic Router and Deepfake Detection pipeline via a web UI.
```bash
python app.py
```
*Navigate to `http://localhost:5000` in your browser. Real-time processing logs will be streamed via WebSockets as your media is analyzed.*

### 2. Analyzing a Video via CLI
This pipeline splits the video into frames, detects faces, runs the model, and gives a final Real/Fake score.

```bash
python -m src.video.parallel_processor \
  --video "path/to/suspect_video.mp4" \
  --model "dsmpe_net_final.pth" \
  --visualize
```
*   **Output:** It will print the score and save a `debug_patches.png` showing what it saw.

### 3. Analyzing a Single Image via CLI
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
