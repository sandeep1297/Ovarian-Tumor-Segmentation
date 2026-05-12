# Swin-NeXt for Ovarian Tumor Ultrasound Segmentation (MMOTU Dataset)

## Overview

This project implements an advanced **Swin Transformer + ConvNeXt hybrid segmentation architecture** for **ovarian tumor ultrasound image segmentation** using the **MMOTU dataset**.

The model is designed for high-accuracy medical image segmentation and includes:

* **Custom Swin Transformer Encoder**
* **ConvNeXt-enhanced Decoder**
* **CBAM Bottleneck Attention**
* **Attention Gates for Skip Connections**
* **Deep Supervision Outputs**
* **Advanced Data Augmentation**
* **Mixed Precision Training (AMP)**
* **Gradient Accumulation**
* **Cosine Annealing + Warm Restarts Scheduler**
* **Post-processing with Morphological Operations**
* **Comprehensive Evaluation Metrics**

---

## Key Features

### Architecture Components

### Encoder

* Patch Embedding Layer
* Window-based Multi-head Self Attention (W-MSA)
* Shifted Window Attention
* Relative Position Bias
* Patch Merging Layers
* DropPath Regularization

### Bottleneck

* CBAM (Channel + Spatial Attention)
* Residual Refinement

### Decoder

* ConvNeXt Blocks
* Attention Gates
* Multi-scale Feature Fusion
* Deep Supervision Heads

---

## Performance Goals

Optimized for:

* High Dice Score
* High mIoU
* Stable validation performance
* Reduced overfitting
* Better boundary segmentation

---

## Training Pipeline

### Steps Included:

1. Dataset loading
2. Dataset mean/std calculation
3. Strong training augmentations
4. Validation/test preprocessing
5. Model initialization
6. Advanced weight initialization
7. Loss computation with deep supervision
8. Mixed precision training
9. Scheduler updates
10. Early stopping
11. Best model checkpoint saving
12. Final evaluation on test set

---

## Augmentations Used

* Resize
* Horizontal Flip
* Vertical Flip
* Rotation
* Elastic Transform
* Grid Distortion
* CLAHE
* Random Brightness/Contrast
* Gaussian Noise
* Normalization

---

## Evaluation Metrics

The project tracks:

* Loss
* IoU
* mIoU
* Dice Score
* Pixel Accuracy

---

## Post-processing

Predicted masks are refined using:

* Morphological Opening
* Morphological Closing
* Connected Component Analysis
* Hole Filling

---

## Main Files

### Notebook

* `Final_code_14.ipynb`

### Saved Model

* `Final_Optimized_SwinNext.pth`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## requirements.txt

```txt
torch
torchvision
numpy
pandas
matplotlib
opencv-python
albumentations
scikit-learn
scipy
Pillow
tqdm
jupyter
notebook
```

---

## Running the Project

### Train Model

Run all notebook cells sequentially:

```bash
jupyter notebook Final_code_14.ipynb
```

### Test Model

Load saved checkpoint and run testing pipeline cells.

---

## Hardware Recommendations

For optimal performance:

* NVIDIA GPU (8GB+ VRAM recommended)
* CUDA support
* Python 3.9+
* 16GB+ RAM

---

## Hyperparameters (Default)

| Parameter               | Value           |
| ----------------------- | --------------- |
| Image Size              | 448       |
| Batch Size              | Configurable    |
| Optimizer               | AdamW           |
| Learning Rate           | Warmup + Cosine |
| Drop Path Rate          | 0.3             |
| Window Size             | 12              |
| Embed Dim               | 96              |
| Early Stopping Patience | 25              |

---

## Output Visualizations

Generated plots include:

* Training vs Validation Loss
* Dice Curve
* IoU Curve
* Accuracy Curve
* Prediction Samples
* Ground Truth Comparison

---

## Future Improvements

Potential upgrades:

* Pretrained Swin backbone
* Test Time Augmentation (TTA)
* Boundary-aware loss
* Multi-class segmentation
* Knowledge distillation
* Ensemble learning

---

## Author Notes

This implementation is highly optimized for experimentation and publication-level benchmarking on the MMOTU ovarian tumor ultrasound dataset.

---

## License

Use for academic and research purposes.
