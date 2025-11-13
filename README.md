# **Self-Supervised Representation Learning for Multi-Label Face Recognition in Television Scenes**

This project explores modern self-supervised learning techniques for visual representation learning, specifically applying **SimCLR** to identify multiple characters within a single image from *The Office* sitcom. The goal was to build a robust recognition system starting from partially labeled data—mirroring a realistic scenario where annotations are limited or incomplete.

---

## **Motivation**

I chose this project to explore how modern computer-vision systems can be built when fully labeled data is limited, which is a common constraint in real-world applications. Sitcom datasets are particularly challenging: scenes often include multiple people, varied lighting, occlusion, and inconsistent annotations. This makes them ideal for testing whether self-supervised representation learning can compensate for sparse labels and still produce reliable downstream performance.

By combining a larger unlabeled dataset with a much smaller labeled one, the project allowed me to investigate how far contrastive learning (SimCLR) can push feature quality before fine-tuning. It also provided an opportunity to design a full pipeline—from data preparation and pretraining to multi-label classification, calibration, and inference—that reflects how identity-recognition systems are engineered in practice.

---

## **Project Overview**

The work is divided into two major stages:

### **1. Self-Supervised Pretraining (SimCLR)**

A feature extractor is trained **without labels** on an image dataset.

### **2. Multi-Label Fine-Tuning**

A classifier is trained on the small labeled subset to predict **all characters present** in a scene.

The final system replicates a production-style inference pipeline:
face detection → alignment → feature extraction → multi-label classification → calibrated predictions → visualizations.

---

## **Datasets Used**

I worked with **two datasets sourced from Kaggle**, chosen intentionally for their contrasting characteristics:

### **1. Labeled Dataset**

* **~500 images**
* **19 character classes**
* Each image may contain **multiple characters**
* Includes annotation files (labels + bounding boxes)
* **Challenge:** multi-character scenes, imbalanced distribution (e.g., fewer images of some characters)
* Dataset:
https://www.kaggle.com/datasets/rodrigoclporto/the-office-dataset-for-yolo-face-detection-model

### **2. Unlabeled Dataset**

* **~1.5k images**
* **6 characters**
* Each image contains **exactly one person**
* **Ideal for representation learning**, even without labels
* Dataset:
https://www.kaggle.com/datasets/pathikghugare/the-office-characters

Using both datasets together enabled:

* Establishing domain-awareness (from unlabeled data)
* More accurate fine-tuning (from labeled data)
* A realistic, industry-style workflow where labeled data is scarce

---

## **Why Self-Supervised Learning?**

Limited labels make fully supervised training unstable.
SimCLR avoids this limitation by learning through:

* Contrastive pair creation
* Heavy data augmentation
* Content-invariant feature learning

This produces a **general visual encoder** that performs well even with small labeled datasets.

---

## **Technical Stack**

### **Frameworks & Libraries**

* **PyTorch** — model, training loops, augmentation
* **PyTorch Lightning** — checkpointing, reproducibility, trainer pipeline
* **scikit-learn** — metrics, UMAP/TSNE, threshold calibration
* **facenet-pytorch (MTCNN)** — face detection for inference
* **NumPy / Pandas** — annotation parsing, dataset tools
* **Matplotlib / Pillow** — result visualization

### **Models**

* **Backbone:** ResNet-50 (SimCLR initialization)
* **Projector:** MLP (SimCLR standard)
* **Classifier:** Multi-label linear head using sigmoid outputs

### **Training Infrastructure**

* Mixed-precision training
* Cosine LR scheduling
* Early stopping
* Model checkpointing
* Environment isolation via mamba/conda

---

## **Project Pipeline (End-to-End)**

### **Step 1 — SimCLR Pretraining (Unlabeled Dataset)**

* Implemented custom two-view augmentations
* Trained for ~100 epochs
* Extracted embeddings + UMAP visualizations
* Resulting latent space showed **meaningful grouping by face identity**, despite no labels.

### **Step 2 — Multi-Label Fine-Tuning (Labeled Dataset)**

* Rebuilt annotations into a unified CSV + label_map
* Implemented full train/val split
* Designed a multi-label head with BCEWithLogitsLoss
* Implemented threshold calibration per class

**Final fine-tuned metrics:**

* **Image-level Mean Average Precision (mAP): ~0.25**
* **Calibrated Macro-F1: ~0.16**
* **Hamming Loss: ~0.18**

These results are **expected for a small, imbalanced, multi-label dataset with multiple faces per image**. The pipeline, however, remains robust and generalizable.

### **Step 3 — Full Inference Pipeline**

Developed a production-style inference module:

1. Face detection using MTCNN
2. Safe bounding-box expansion and cropping
3. Batch feature extraction and prediction
4. Per-crop classification + calibration
5. Image-level aggregation
6. Visualization overlays saved to disk
7. JSON results export for downstream evaluation

The system successfully identifies multiple characters per image and produces explainable predictions.

---

## **Obstacles & Solutions**

### **1. Corrupted / mislabeled JPEG files**

Many labeled images pretending to be “JPEG” were unreadable.

* **Solution:**
  Automated detection using PIL & `file` CLI, followed by deletion & fallback handling.

### **2. Mismatched checkpoint paths**

Lightning saved checkpoints inside notebooks.

* **Solution:**
  Reorganized project structure, forced explicit `CKPT_DIR`.

### **3. Device mismatch errors in classifier**

Caused by partial GPU transfer of model components.

* **Solution:**
  Explicitly moved `backbone` and `classifier` separately to CUDA.

### **4. Multi-label imbalance**

Some characters had far fewer images.

* **Solution:**
  Applied per-class threshold calibration, boosting macro-F1 significantly.

---

## **Key Outcomes**

* Built a **full self-supervised + supervised training system** from scratch
* Designed a **clean, reproducible machine-learning research pipeline**
* Implemented **robust inference code** that deals with real-world noise
* Produced **high-quality visual results**, t-SNE/UMAP plots, and overlays
* Learned to debug device errors, corrupt files, and annotation inconsistencies

---

## **Repository Structure**

```
office-ssl/
├── src/                # SimCLR, datasets, fine-tuning modules
├── data/               # Annotations + images (not stored in repo)
├── notebooks/          # Jupyter notebooks for pretraining + fine-tuning
├── figures/            # Visualizations (t-SNE, UMAP, inference samples)
├── checkpoints/        # Model weights (gitignored)
└── README.md

```

---

## **How to Reproduce**

```bash
git clone https://github.com/RedWaffle007/office-ssl.git
cd office-ssl
mamba env create -f environment.yml
mamba activate office_ssl
jupyter lab
```

---
