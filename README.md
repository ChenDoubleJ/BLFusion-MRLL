<div align="center">

# BLFusion-MRLL

### Breaking Low-Light Fusion Barrier: Unsupervised Darkness and Noise-Aware Visible and Infrared Image Fusion Network 

### Multi-modal Real-world Low-light Dataset

[![Paper](https://img.shields.io/badge/Paper-TIP(CCF--A)%20Accepted-blue)](#citation)
[![Code](https://img.shields.io/badge/Code-available-brightgreen)](#code-availability)
[![Dataset](https://img.shields.io/badge/MRLL(Dataset)-available-brightgreen)](#dataset-availability)

Official repository for **BLFusion** and the **MRLL** dataset.

</div>

---

## 📌 Overview

**BLFusion** is an unsupervised darkness- and noise-aware infrared-visible image fusion framework for challenging nighttime scenes. It is designed to jointly handle two major low-light barriers: insufficient illumination and complex real-world noise.

<p align="center">
  <img width="861" alt="BLFusion overview" src="https://github.com/user-attachments/assets/facb4247-a1f4-40e5-bfac-b2c34bcb44ab" />
</p>

Existing low-light fusion methods often focus on brightness enhancement while overlooking noise suppression. Degradation-aware supervised methods can model multiple degradations, but they usually rely on paired high-quality references and may suffer from imbalanced optimization between enhancement and denoising.

BLFusion addresses these issues with a customized two-stage unsupervised framework. The first stage enhances dark visible images through Retinex-guided decomposition with state space modeling. The second stage performs denoising and fusion jointly, producing fused images with clearer structures, richer textures, and stronger robustness under real low-light noise.

## ✨ Highlights

- **Darkness- and noise-aware fusion**: simultaneously improves illumination and suppresses complex noise in low-light infrared-visible fusion.
- **Retinex-guided SSM decomposition**: combines Retinex theory with state space modeling to enhance dark visible images while preserving structural details.
- **Unsupervised denoising fusion**: integrates pixel-shuffle downsampling, blind-spot networks, and dilated convolutions to learn noise-free features without clean references.
- **Maximum replacement refinement**: reduces visual artifacts and further improves fused image quality during inference.
- **MRLL dataset**: provides 500 aligned infrared-visible image pairs captured from real-world indoor and outdoor nighttime scenes.

## 🧩 Framework

<p align="center">
  <img width="100%" alt="BLFusion framework" src="https://github.com/user-attachments/assets/2039acd6-55c0-4f1a-9d20-409325c96f8f" />
</p>

BLFusion contains two dedicated stages:

1. **State Space Model-Based Decomposition Network**: decomposes low-light visible images into illumination and reflectance components. The Retinex-inspired design models illumination degradation, while efficient state space modules capture long-range dependencies and preserve fine visual details.
2. **Unsupervised Denoising and Fusion Network**: extracts noise-suppressed features from enhanced visible and infrared images. Pixel-shuffle downsampling disrupts spatial noise correlation, blind-spot networks avoid identity mapping, and dilated convolutions expand contextual perception for robust denoising and fusion.

## 🗓️ Release Plan

The full project will be released after the official paper process is complete. Planned materials include:

- [x] Source code for the official BLFusion implementation
- [x] Pretrained model checkpoints
- [x] Testing and evaluation scripts
- [x] Training scripts and configuration files
- [x] MRLL dataset download links
- [x] Visualization tools and qualitative comparison results
- [x] Detailed usage instructions and reproducibility notes

## 🛠️ Installation

Create a conda environment named **BLFusion** and install the core dependencies:

```bash
conda create -n BLFusion python=3.8 -y
conda activate BLFusion

pip install torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt
```

The project uses Mamba/SSM modules. If `mamba-ssm` or `causal-conv1d` fails to build, please make sure the local CUDA toolkit and PyTorch CUDA version are compatible before reinstalling the requirements. I previously referred to the following tutorial for installation guidance: https://blog.csdn.net/qq_45100200/article/details/139754231.

## 📦 Pretrained Models

Pretrained checkpoints are available from Baidu Drive:

| Source | Link | Access Code |
|---|---|---|
| Baidu Drive | [Download](https://pan.baidu.com/s/1ZFyoQ-ZVuR_vrb50_fT9vA) | `yvkz` |

After downloading `BLFusion_weights`, place the checkpoint files into the corresponding folders:

```text
Enhancement_Network/
└── weights/
    └── checkpoint_Diff_TDN.pth

Fused_Denoise_Network/
└── weights/
    ├── double_noise/
    │   └── checkpoint_Diff_TDN.pth
    ├── mul_noise/
    │   └── checkpoint_Diff_TDN.pth
    ├── no_noise/
    │   └── checkpoint_Diff_TDN.pth
    └── ran_noise/
        └── checkpoint_Diff_TDN.pth
```

These paths match the default checkpoint locations used by the provided testing scripts.

## 🧪 Testing

BLFusion follows a two-stage testing workflow. First, run the enhancement network to generate enhanced visible images. Then copy the enhancement results into the fusion network test set and run the denoising-fusion network.

### Stage 1: Visible Image Enhancement

Enter the enhancement network folder:

```bash
cd Enhancement_Network
```

Check `test.py` and modify the dataset path, checkpoint path, and result-saving path if needed:

```python
read_data("dataset/MRLL")
model_weight_path = "./weights/checkpoint_Diff_TDN.pth"
root = "./results/MRLL"
```

Run the enhancement script:

```bash
python test.py
```

The enhancement results will be saved under:

```text
Enhancement_Network/results/MRLL/
```

### Stage 2: Denoising and Fusion

Copy the enhanced visible results to the test directory used by the fusion network. For example:

```bash
cd ..
cp -r Enhancement_Network/results/MRLL/vi_l Fused_Denoise_Network/dataset/MRLL/test/
cp -r Enhancement_Network/results/MRLL/vi_r Fused_Denoise_Network/dataset/MRLL/test/
cp -r Enhancement_Network/results/MRLL/vi Fused_Denoise_Network/dataset/MRLL/test/
cp -r Enhancement_Network/results/MRLL/ir Fused_Denoise_Network/dataset/MRLL/test/
```

Then enter the fusion network folder:

```bash
cd Fused_Denoise_Network
```

Check `test.py` and modify the dataset path, checkpoint path, and fusion result-saving path if needed:

```python
read_data_test("dataset/MRLL")
model_weight_path = "./weights/mul_noise/checkpoint_Diff_TDN.pth"
root = "./results/MRLL/"
```

Select the checkpoint according to the target noise setting:

```text
weights/double_noise/checkpoint_Diff_TDN.pth  # both visible and infrared modalities contain multiplicative noise
weights/mul_noise/checkpoint_Diff_TDN.pth     # visible images contain multiplicative noise
weights/no_noise/checkpoint_Diff_TDN.pth      # no noise setting
weights/ran_noise/checkpoint_Diff_TDN.pth     # random noise setting
```

Run the fusion script:

```bash
python test.py
```

The fused images will be saved to the result directory specified in `test.py`, for example:

```text
Fused_Denoise_Network/results/MRLL/fused/
```

## 🚀 Training

Similar to Testing, run `python train.py` !

---

# MRLL Dataset

**MRLL** stands for **Multi-modal Real-world Low-Light Dataset**. It is a synchronized dual-spectral dataset built for infrared-visible image fusion under realistic nighttime degradation.

MRLL contains **500 well-aligned infrared-visible image pairs** with a resolution of **960 x 720**. The dataset covers diverse indoor and outdoor low-light scenes, varying illumination levels, and complex real-world noise patterns, making it suitable for evaluating fusion models under practical nighttime conditions.

## 📥 Dataset Availability

The full MRLL dataset is available at:

| Source | Link | Access Code |
|---|---|---|
| Baidu Drive | [Download](https://pan.baidu.com/s/1jEnsi1v-xEGYtOvUQ6Oj9w?pwd=ek7j) | `ek7j` |
| Google Drive | [Download](https://drive.google.com/drive/folders/1Yu_6_Z0zjqYkV4GfWAu8ZtW88bn77yWN?usp=drive_link) | - |

## 🔍 Synchronized Dual-Spectral Imaging and Registration

MRLL is collected with synchronized visible and infrared imaging devices (DJI Matrice 4T). The visible and infrared images are aligned through cross-modal registration to obtain paired dual-spectral samples for fusion research.

<p align="center">
  <img width="100%" alt="Synchronized dual-spectral imaging and registration" src="https://github.com/user-attachments/assets/44d23316-6587-41e0-aa77-abb1115bd0ef" />
</p>

## 🌃 Scenario Schematic

MRLL covers diverse real-world low-light scenarios, including indoor and outdoor nighttime scenes with weak illumination, complex lighting, texture degradation, and challenging visibility.

<p align="center">
  <img width="100%" alt="Scenario schematic" src="https://github.com/user-attachments/assets/06983185-f1d2-4923-8e5b-8fc1fe67c4bb" />
</p>

## 📊 Comparison with Existing Low-Light Fusion Datasets

Compared with existing low-light infrared-visible fusion datasets, MRLL provides synchronized real-world paired data, indoor and outdoor coverage, and multiple realistic noise patterns.

<p align="center">
  <img width="100%" alt="Comparison of low-light fusion datasets" src="https://github.com/user-attachments/assets/f5a8c870-52ee-406b-94cf-a77cc484af48" />
</p>

<p align="center">
  <img width="900" alt="Dataset comparison details" src="https://github.com/user-attachments/assets/20556f56-9115-430a-bc3b-90ee5e802e70" />
</p>


## 📚 Citation

If you find this work or dataset useful for your research, please consider citing our paper:

```bibtex
@article{chen2026breaking, 
   title={Breaking Low-Light Fusion Barrier: Unsupervised Darkness and Noise-Aware Visible and Infrared Image Fusion Network},
   author={Chen, Jiajun and Xu, Han and Liu, Guangcan and Ma, Jiayi},
   journal={IEEE Transactions on Image Processing},
   year={2026},
   publisher={IEEE}
}
```
## 📬 Contact

For questions about BLFusion or the MRLL dataset, please open an issue in this repository.
