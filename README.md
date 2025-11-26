[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17681654.svg)](https://doi.org/10.5281/zenodo.17681654)
# Dynamic Adaptive Network for Precise Camouflaged Object Detection (DABNet)
This repository provides the official PyTorch implementation of the paper **"Dynamic Adaptive Network for Precise Camouflaged Object Detection"**, currently submitted to *The Visual Computer*. A permanent archived version of this code is available at: https://doi.org/10.5281/zenodo.17681654. Readers are reminded that this repository is directly related to the manuscript, and we encourage you to cite the paper when using this code or related results.
## 📝 Overview
DABNet is a dynamic-adaptive and boundary-aligned network designed for precise Camouflaged Object Detection (COD). The framework integrates a Dynamic Adaptive Scale Module (DASM) for content-aware multi-scale modeling, and a Boundary-guided Deformable BiFPN (BD-BiFPN) for structure-consistent deformable alignment. The model improves scale adaptability, boundary fidelity, and robustness in weak-contrast camouflage scenarios.
## 📦 Environment Setup
### 1. Create Conda Environment
```bash
conda create -n DABNet python=3.8
conda activate DABNet
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## 📁 Dataset Preparation
Download COD datasets from: https://github.com/visionxiang/awesome-camouflaged-object-detection?tab=readme-ov-file#Datasets
### Training Set Structure
```
../TrainDataset/
├── Imgs/
│     ├── 0001.jpg
│     └── ...
└── GT/
      ├── 0001.png
      └── ...
```
### Testing Set Structure
```
../TestDataset/
├── CAMO/
│     └── Test/Imgs + GT
├── COD10K/
│     └── Test/Imgs + GT
└── NC4K/
      └── Test/Imgs + GT
```
## 🚀 Training
```bash
python Train.py --epoch 40 --lr 1e-4 --batchsize 4 --trainsize 704 --train_path ../TrainDataset --save_path ./checkpoints/
```
Models will be saved under `./checkpoints/`.
## 🧪 Testing
```bash
python Test.py --testsize 704 --pth_path ./checkpoints/model_epoch_40.pth --test_path ../TestDataset/
```
Prediction maps will be saved under:
```
./results/
├── CAMO/
├── COD10K/
└── NC4K/
```
## 📊 Evaluation
```bash
python eval.py --model DABNet --pred_root ./results --GT_root ../TestDataset --record_path output/eval_record.txt --datasets CAMO COD10K NC4K --BR on
```
Results will be stored in `output/eval_record.txt`.
## 📚 Project Structure
```
.
├── DABNet.py
├── Train.py
├── Test.py
├── eval.py
├── metrics.py
├── requirements.txt
├── checkpoints/
├── results/
└── datasets/
```
## 📝 Citation
If you use this repository, please cite:
```bibtex
@article{Guo2025DABNet,
  title   = {Dynamic Adaptive Network for Precise Camouflaged Object Detection},
  author  = {Xu Guo and Jie Liu and Yixiao Sun and Wenyu Zhang and Yan Liu and Xiaomeng Liu},
  journal = {The Visual Computer},
  year    = {2025}
}
```
## 📄 License
This project is released under the **MIT License**.
