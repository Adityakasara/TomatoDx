# TomatoDx: Deep Learning–Based Tomato Leaf Disease Identification

> **Pre-Final Year Engineering Project** | Computer Vision & Agricultural Informatics  
> Model: EfficientNetB0 Transfer Learning | Framework: TensorFlow/Keras

---

## Abstract

Tomato (*Solanum lycopersicum*) is one of the world's most economically important crops. Fungal, bacterial, and viral diseases cause yield losses of up to 40% annually. Traditional disease diagnosis relies on expert agronomists, which is costly and inaccessible to smallholder farmers. This project presents **TomatoDx**, a deep learning system that automatically identifies 10 tomato leaf disease categories (plus healthy) from smartphone-captured images with >95% accuracy. The system uses EfficientNetB0 pretrained on ImageNet and fine-tuned on the PlantVillage dataset, with Grad-CAM explainability to highlight discriminative leaf regions.

---

## Project Structure

```
TomatoDx/
├── TomatoDx_Project.ipynb      # Main notebook (primary deliverable)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── tomatodx_train.py       # Standalone training script
│   ├── tomatodx_inference.py   # Inference on new images
│   └── tomatodx_gradcam.py     # Grad-CAM visualization utilities
├── data/
│   └── tomato/                 # Dataset root (see Dataset Setup below)
│       ├── train/
│       ├── val/
│       └── test/
└── outputs/
    ├── tomatodx_best.keras     # Saved best model
    ├── training_history.png    # Accuracy/loss curves
    └── confusion_matrix.png    # Evaluation heatmap
```

---

## Dataset Setup

This project uses the **PlantVillage Tomato** subset from Kaggle.

### Option A — Kaggle API (Recommended)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your kaggle.json API token at ~/.kaggle/kaggle.json
#    (Download from: https://www.kaggle.com/settings → API → Create New Token)

# 3. Download and extract
kaggle datasets download -d emmarex/plantdisease -p ./data --unzip

# 4. The notebook will automatically filter tomato classes and split into train/val/test
```

### Option B — Manual Download

1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download `PlantVillage.zip`
3. Extract to `./data/PlantVillage/`
4. Run the dataset preparation cell in the notebook

---

## Tomato Disease Classes (11 Classes)

| # | Class | Type |
|---|-------|------|
| 0 | Tomato_Bacterial_spot | Bacterial |
| 1 | Tomato_Early_blight | Fungal |
| 2 | Tomato_Late_blight | Oomycete |
| 3 | Tomato_Leaf_Miner | Insect |
| 4 | Tomato_Leaf_Mold | Fungal |
| 5 | Tomato_Septoria_leaf_spot | Fungal |
| 6 | Tomato_Spider_mites | Arachnid |
| 7 | Tomato_Target_Spot | Fungal |
| 8 | Tomato_Tomato_Yellow_Leaf_Curl_Virus | Viral |
| 9 | Tomato_Tomato_mosaic_virus | Viral |
| 10 | Tomato_healthy | Healthy |

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd /path/to/TomatoDx

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook TomatoDx_Project.ipynb
# Run: Kernel → Restart & Run All
```

### Command-Line Training

```bash
python src/tomatodx_train.py \
    --data_dir ./data/tomato \
    --epochs 40 \
    --batch_size 32 \
    --output_dir ./outputs
```

### Inference on a New Image

```bash
python src/tomatodx_inference.py \
    --model_path ./outputs/tomatodx_best.keras \
    --image_path /path/to/leaf.jpg
```

### Grad-CAM Visualization

```bash
python src/tomatodx_gradcam.py \
    --model_path ./outputs/tomatodx_best.keras \
    --image_path /path/to/leaf.jpg \
    --output_path ./outputs/gradcam_overlay.png
```

---

## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet pretrained, base frozen in Phase 1)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(11, activation='softmax')   ← Output: 11 disease classes
```

**Training Strategy:**
- **Phase 1** (20 epochs): Freeze EfficientNetB0 base, train classification head only. Adam lr=1e-3.
- **Phase 2** (20 epochs): Unfreeze top-30 layers of base for fine-tuning. Adam lr=1e-5.
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5), ModelCheckpoint (best val_accuracy).

---

## Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95–97% |
| Macro Precision | ~95% |
| Macro Recall | ~95% |
| Macro F1-Score | ~95% |

*(Exact values depend on dataset version and hardware. See notebook for full classification report.)*

---

## Future Scope

- Deploy as a mobile app (TensorFlow Lite conversion)
- Extend to multi-crop disease detection
- Integrate severity estimation (mild / moderate / severe)
- Real-time detection using drone imagery

---

## References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images for identification of plant diseases. *arXiv:1511.08060*.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML 2019*.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV 2017*.
4. Mohanty, S. P., et al. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*.

---

## License

This project is submitted as an academic deliverable. Dataset usage is subject to the PlantVillage dataset license.
