# Advanced Deepfake Detection Framework

> **Dual-Stream Multi-Patch Ensemble Network (DSMPE-Net)**
> A robust, explainable, and high-performance deepfake detection system designed for both image and video forensics.


## 🔍 Overview

This is a novel dual-stream architecture to detect manipulation artifacts in digital media. Unlike traditional methods that only look at visual anomalies, DeepScan analyzes the **Frequency Domain (DCT)** to catch generative artifacts invisible to the naked eye.

### Key Features
*   **Dual-Stream Architecture**: Combines **Spatial (Xception)** and **Frequency (ResNet50+DCT)** streams.
*   **Multi-Patch Ensemble**: Splits images into 9 overlapping patches to detect local inconsistencies.
*   **Smart Video Pipeline**: 
    *   **Filters**: Automatically skips blurry or duplicate frames.
    *   **Parallel Processing**: Decodes video on CPU while running Inference on GPU.
    *   **Visualization**: See exactly what the model sees.
*   **Explainability**: Provides patch-level confidence scores to pinpoint manipulated regions.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU (Recommended for video inference)

### Setup
1.  **Clone the repository**
    ```bash
    git clone https://github.com/MohithTP/DeepFake.git
    cd DeepFake
    ```

2.  **Install Dependencies**
    Using `uv` (recommended) or pip:
    ```bash
    uv pip install -r requirements.txt
    # OR
    pip install -r requirements.txt
    ```

3.  **Download Weights**
    Place your trained model weights (e.g., `dsmpe_net_epoch_1.pth`) in the project root.


---

## 📂 Datasets

This framework is compatible with major deepfake forensics datasets.

### 1. Celeb-DF v2 (Recommended)
*   **Description**: High-quality deepfake videos featuring celebrities. Used for our primary **Video Inference** testing to ensure robustness against high-quality generations.
*   **Why**: Overcomes the saturation issues of older datasets like FaceForensics++.
*   **Structure**:
    ```
    Celeb-DF-v2/
    ├── Celeb-real/
    ├── Celeb-synthesis/
    └── YouTube-real/
    ```

### 2. WildDeepfake (Training)
*   **Description**: Diverse collection of real-world deepfakes found on the internet.
*   **Why**: Used for training the core model to generalize against "in-the-wild" variations.

---


## 🚀 Usage

### 1. Video Analysis (Recommended)
Run the high-performance parallel processor on any MP4 file.

```bash
uv run python -m src.video.parallel_processor --video "path/to/video.mp4" --model dsmpe_net_epoch_1.pth --visualize
```

**Arguments:**
*   `--video`: Path to input video.
*   `--model`: Path to model weights.
*   `--blur_thresh`: (Optional) Blur sensitivity (low value = allow more frames). Default: 5.0.
*   `--visualize`: (Optional) Save the 9-patch grid to `debug_patches.png`.

### 2. Single Image Inference
Analyze a specific image frame.

```bash
uv run python inference.py --image_path "path/to/image.jpg" --weight_path dsmpe_net_epoch_1.pth
```

### 3. Training
This project is optimized for training on Kaggle/Colab due to dataset sizes.
*   See `kaggle_guide.md` for full training instructions.
*   **Resume Training**:
    ```bash
    python train_image.py --resume_path dsmpe_net_epoch_1.pth
    ```

---

## 🧠 Architecture Details

### DSMPE-Net Structure
1.  **Input**: $1024 \times 1024$ Image/Frame.
2.  **Patch Extraction**: Splits input into **9** overlapping $256 \times 256$ patches.
3.  **Stream 1 (Spatial)**: `Xception` backbone extracts visual features.
4.  **Stream 2 (Frequency)**: `ResNet50` processes **Discrete Cosine Transform (DCT)** coefficients.
5.  **Fusion**: Concatenates features ($2048 + 2048 = 4096$ dim per patch).
6.  **Meta-Classifier**: Aggregates all 9 patch scores into a final **Real vs. Fake** verdict.

---

## 🚧 Roadmap

*   [x] Image Inference Pipeline
*   [x] Smart Video Processing (Parallelized)
*   [ ] **Face Detection Integration (MTCNN)** - *Coming Soon*
*   [ ] Audio Deepfake Detection
*   [ ] Real-time Webcam Inference

---

