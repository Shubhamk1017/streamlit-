# Crack Segmentation using U-Net
**Shubham Kumar** | 25B0674 | Civil Engineering, IIT Bombay  
Summer of Science 2026 — AI/ML Track

## Overview
This project implements a **U-Net** deep learning model for pixel-level crack detection in concrete and pavement surfaces. Automated crack detection is important for structural health monitoring — instead of manual inspection (which is slow and error-prone), a trained model can quickly identify cracks in images of buildings, bridges, and roads.

## Architecture
I built the U-Net from scratch in PyTorch following the [original paper](https://arxiv.org/abs/1505.04597) by Ronneberger et al. (2015). The key ideas:
- **Encoder** gradually reduces spatial resolution while learning features (what cracks look like)
- **Decoder** upsamples back to full resolution for pixel-level prediction
- **Skip connections** pass spatial details from encoder to decoder (crucial for thin crack structures)

```
Input (3, 256, 256)
    ├── Encoder: 64 → 128 → 256 → 512
    ├── Bottleneck: 1024
    ├── Decoder: 512 → 256 → 128 → 64
    └── Output (1, 256, 256)
```

Total parameters: ~31 million

## Dataset
[Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) from Kaggle:
- ~11,200 images with pixel-level binary masks
- Merged from 12 different crack image sources
- Resolution: 448×448 (resized to 256×256 for training)

## Project Structure
```
Crack Segmentation/
├── crack_segmentation.py   ← Main training script (Colab-compatible)
├── unet.py                 ← U-Net architecture from scratch
├── predict.py              ← Inference on new images
├── requirements.txt        ← Dependencies
└── README.md               ← You are here
```

## How to Run

### On Google Colab (Recommended)
1. Upload `crack_segmentation.py` to Google Colab
2. The script will prompt you to download the dataset using Kaggle API
3. Run all cells top to bottom

### Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle, extract to ./crack_segmentation_dataset/

# Run training
python crack_segmentation.py

# Run predictions on new images
python predict.py --image path/to/crack_photo.jpg
python predict.py --dir path/to/images/ --output results/
```

## Training Details
- **Loss:** BCE + Dice Loss (handles class imbalance — cracks are sparse pixels)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau (halves LR when validation loss plateaus)
- **Early stopping:** patience=5 epochs
- **Augmentation:** Random flips, rotations, brightness/contrast
- **Image size:** 256×256

## Evaluation Metrics
- **IoU (Intersection over Union)** — primary segmentation metric
- **Dice Coefficient** — F1-score equivalent for segmentation
- **Pixel Accuracy, Precision, Recall**

## Output Files
After training:
- `best_model.pth` — best model weights (by validation loss)
- `crack_segmentation_final.pth` — full checkpoint with training history
- `training_curves.png` — loss and dice plots
- `predictions.png` — sample test predictions
- `overlays.png` — crack detection overlays on original images
- `sample_data.png` — dataset visualization

## References
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Original dataset: [github.com/khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation)
- Andrew Ng, CS229 & Deep Learning Specialization
- StatQuest, CampusX YouTube channels

## Acknowledgments
Mentor: Sagnik Dey  
Summer of Science 2026, IIT Bombay
