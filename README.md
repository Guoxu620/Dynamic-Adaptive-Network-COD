[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17680913.svg)](https://doi.org/10.5281/zenodo.17680913)

This repository provides the official implementation of the paper
**"Dynamic Adaptive Network for Precise Camouflaged Object Detection"**,
which is currently submitted to *The Visual Computer*.

The archived version of this code is permanently available at
https://doi.org/10.5281/zenodo.17680913. Readers are kindly reminded that this
repository is directly related to the above manuscript, and we encourage you to
cite the manuscript when using this code or related results.
# 📦 1. Environment Setup

### **1.1 Create Conda Environment**

```
conda create -n DABNet python=3.8
conda activate DABNet
```

### **1.2 Install Dependencies**

```
pip install -r requirements.txt
```

------

# 📁 2. Dataset Preparation

Download COD datasets from:
 🔗 https://github.com/visionxiang/awesome-camouflaged-object-detection?tab=readme-ov-file#Datasets

------

## **2.1 Training Set Structure**

Your training dataset must follow this structure:

```
../TrainDataset/
│
├── Imgs/
│     ├── 0001.jpg
│     ├── 0002.jpg
│     └── ...
│
└── GT/
      ├── 0001.png
      ├── 0002.png
      └── ...
```

- `Imgs/` — training images
- `GT/` — corresponding ground-truth binary masks

------

## **2.2 Testing Set Structure**

Each testing dataset (CAMO, COD10K, NC4K) must follow:

```
../TestDataset/
│
├── CAMO/
│     └── Test/
│           ├── Imgs/
│           └── GT/
│
├── COD10K/
│     └── Test/
│           ├── Imgs/
│           └── GT/
│
└── NC4K/
      └── Test/
            ├── Imgs/
            └── GT/
```

This structure is required for both `Test.py` and `eval.py`.

------

# 🚀 3. Training

Example training command (40 epochs):

```
python Train.py \
    --epoch 40 \
    --lr 1e-4 \
    --batchsize 4 \
    --trainsize 704 \
    --train_path ../TrainDataset \
    --save_path ./checkpoints/
```

Trained models will be saved in:

```
./checkpoints/
```

------

# 🧪 4. Testing

Generate prediction maps for each dataset:

```
python Test.py \
    --testsize 704 \
    --pth_path ./checkpoints/model_epoch_40.pth \
    --test_path ../TestDataset/
```

Prediction results will be saved under:

```
./results/
│
├── CAMO/
├── COD10K/
└── NC4K/
```

------

# 📊 5. Evaluation

Evaluate predictions using standard COD metrics (S-measure, MAE, E-measure, F-measure):

```
python eval.py \
    --model RISNet \
    --pred_root ./results \
    --GT_root ../TestDataset \
    --record_path output/eval_record.txt \
    --datasets CAMO COD10K NC4K \
    --BR on
```

Evaluation results will be stored in:

```
output/eval_record.txt
```

------

# 📚 6. Project Structure

```
.
├── DABNet.py
├── Train.py
├── Test.py
├── eval.py
├── requirements.txt
├── checkpoints/
├── results/
└── datasets/
```
