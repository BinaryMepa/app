# 🛰️ Satellite Image Segmentation with U-Net

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

A web app for **semantic segmentation of satellite images**, built with a custom **U-Net** in PyTorch and served with **Streamlit**.
Upload a satellite image and the model predicts a color-coded mask with **19 land-use classes**, such as buildings, roads, water, forests and farmland.

This project was made as a final project for **Deep Learning School (DLS)**.

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model](#-model)
- [Classes](#-classes)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## ✨ Features

- 🧠 **Custom U-Net** with 5 encoder/decoder levels (64 → 2048 channels)
- 🖼️ **Image upload** in JPG, PNG or TIFF format, including grayscale, transparent and 16-bit images
- 📐 **Any image size:** small images are upscaled, large ones are downscaled, and the mask keeps the original aspect ratio
- 🎨 **Color-coded mask** displayed side by side with the original image
- 🗺️ **Class legend** in the sidebar
- ⚡ **Cached model loading**, so the weights load only once

---

## 📁 Project Structure

```text
app/
├── app.py                   # Streamlit web app (entry point)
├── src/
│   ├── __init__.py
│   ├── config.py            # Paths, image size, normalization settings
│   ├── classes.py           # The 19 classes: names and colors
│   ├── model.py             # U-Net architecture
│   ├── image_utils.py       # Safe image loading (size limits, format conversion)
│   ├── inference.py         # Model loading, preprocessing, prediction
│   └── visualization.py     # Turning masks into color images
├── Untitled1.ipynb          # Training / experiments notebook
├── segmentation_model.pth   # Trained weights (Git LFS, ~500 MB)
├── requirements.txt
├── .gitattributes
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- [Git LFS](https://git-lfs.com/). The model weights (`segmentation_model.pth`, about 500 MB) are stored with Git LFS.

### Installation

```bash
# 1. Install Git LFS (once per machine)
git lfs install

# 2. Clone the repository (this also downloads the model weights)
git clone https://github.com/BinaryMepa/app.git
cd app

# 3. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

> 💡 If `segmentation_model.pth` is only a few hundred bytes, it is just an LFS pointer. Run `git lfs pull` to download the real weights.

---

## 🖥️ Usage

Start the app:

```bash
streamlit run app.py
```

Then open <http://localhost:8501> in your browser and:

1. Upload a satellite image (`.jpg`, `.png` or `.tif`).
2. Compare the original image with the predicted mask.
3. Use the sidebar legend to see which color is which class.

### Using the model in Python

```python
from PIL import Image

from src.config import MODEL_PATH
from src.inference import load_model, predict_mask
from src.visualization import colorize_mask

model = load_model(MODEL_PATH)
image = Image.open("example.png")

mask = predict_mask(model, image)      # (H, W) array of class ids, same size as the image
colored = colorize_mask(mask)          # (H, W, 3) RGB image
Image.fromarray(colored).save("mask.png")
```

---

## 🧠 Model

| Component | Details |
| --- | --- |
| Architecture | Custom U-Net with skip connections |
| Encoder | `DoubleConv(3→64)` followed by 5 `Down` blocks: 64 → 128 → 256 → 512 → 1024 → 2048 |
| Decoder | 5 `Up` blocks with transposed convolutions: 2048 → 1024 → 512 → 256 → 128 → 64 |
| Output | 1×1 convolution to 19 class logits |
| Input | RGB image resized to **512 × 512**, normalized with ImageNet mean/std |
| Output mask | Class ids (argmax over logits), resized back to the image's aspect ratio |
| Framework | PyTorch |

---

## 🎨 Classes

| ID | Class | Color |
| :-: | --- | --- |
| 0 | Background / Unclassified | `#DCDCDC` |
| 1 | Residential buildings | `#FFB6C1` |
| 2 | Commercial buildings | `#90EE90` |
| 3 | Roads | `#ADD8E6` |
| 4 | Water bodies | `#98FB98` |
| 5 | Forests | `#DDA0DD` |
| 6 | Farmland | `#E0FFFF` |
| 7 | Vacant land | `#FA8072` |
| 8 | Transport infrastructure | `#D3D3D3` |
| 9 | Parks and recreation areas | `#FFDAB9` |
| 10 | Industrial areas | `#FFFF00` |
| 11 | Bridges | `#00FFFF` |
| 12 | Railways | `#FF00FF` |
| 13 | Airports | `#808000` |
| 14 | Sports facilities | `#008080` |
| 15 | Beaches | `#800080` |
| 16 | Mountains and hills | `#00FF7F` |
| 17 | Rivers and canals | `#FF6347` |
| 18 | Parking lots | `#0000FF` |

Class names and colors are defined in one place: [`src/classes.py`](src/classes.py).

---

## ⚠️ Limitations

- **Image size:** images up to about **7000 × 7000** pixels are accepted. Images larger than 2048 px are downscaled before processing. Images smaller than 512 px work, but are upscaled, so predictions may be less accurate.

- **Accuracy:** the training images did not contain enough information for the model to learn well.
  As a result, the current **Dice score is about 0.08**, so the predicted masks should be treated as a **proof of concept**, not as production-quality results.

---

## 🗺️ Roadmap

- [ ] Improve the training dataset (more images, better masks)
- [ ] Improve the architecture (for example, a pretrained encoder)
- [ ] Add evaluation metrics (Dice, IoU) per class
- [x] Resize the predicted mask back to the original image size
- [ ] Deploy the app to Streamlit Community Cloud

---

## 👤 Author

**BinaryMepa**, [GitHub profile](https://github.com/BinaryMepa)

Made as a final project for Deep Learning School (DLS).
