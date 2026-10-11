# AgriSpectra.AI  (paper-faithful version)

Implements the architecture of "Multimodal Crop Health Monitoring Using Hyperspectral Imaging, IoT Sensors and Deep Learning":
CNN (hyperspectral) + LSTM (IoT) branches with late fusion, Grad-CAM, and a Streamlit dashboard.
`train.py` MEASURES results and prints them next to the paper's reported numbers.

## Setup (Python 3.10-3.12)
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt

## Data
    archive/train/Health/*.tif
    archive/train/Rust/*.tif
    archive/train/Other/*.tif

## Run
    python train.py --epochs 40                 # CNN + LSTM + fusion
    python train.py --epochs 40 --baselines     # + SVM and RGB-CNN baselines
    python train.py --img_size 64               # faster: no 64->128 upsampling
    streamlit run app.py

## Deviations from the paper (be explicit about these when reporting)
- Split is 70/10/20 train/val/test (stratified); the val part is used for early stopping.
- Normalisation is per-cube robust min-max (1-99 percentile), not per-band, to keep band ratios intact.
- Augmentation: flips, 90-degree rotations, brightness (not 15-degree rotation).
- Dataset files are 64x64; `--img_size 128` upsamples (no new information).
- IoT data: exactly as in the paper (1000 uniform records, label rule, T=10, chronological 80/20).
  The rule-based label makes this benchmark almost trivial.
- Fusion: no real image-sensor pairs exist. Sensor windows are SYNTHETICALLY paired to image classes
  (iot_data.make_paired_windows). The fusion score demonstrates the architecture only; disclose this.
- 3D-CNN baseline is not implemented.
