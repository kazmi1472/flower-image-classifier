# Flower Recognition

Image classifier for the Kaggle [Flower Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) dataset — 5 classes (daisy, dandelion, rose, sunflower, tulip), ~4.3k images.

This is my take-home for an AI/ML engineer assessment. It covers EDA, training, evaluation/error analysis and a small inference service.

## Docker

The fastest way to run the inference API — no Python environment needed.

**Prerequisites:** Docker and Docker Compose installed.

**1. Clone the repo**
```bash
git clone https://github.com/kazmi1472/flower-image-classifier.git
cd flower-image-classifier
```

**2. Build and start**
```bash
docker compose up --build
```

**3. Open the interactive docs**
```
http://localhost:8000/docs
```
Click **POST /predict → Try it out → Choose File** → pick any image from `samples/` → **Execute**.

**4. Or call the API directly**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/flo1.jpg" \
  -F "top_k=3"
```

**5. Health check**
```bash
curl http://localhost:8000/health
```

**Stop the container**
```bash
docker compose down
```

> The `models/` directory is mounted as a volume, so you can swap in a retrained model without rebuilding the image — just restart the container.

## Python Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Drop the dataset under `data/flowers/<class>/*.jpg`. The folders should be `daisy/ dandelion/ rose/ sunflower/ tulip/`.

---

## Layout

```
configs/default.yaml      # all knobs live here
data/flowers/             # dataset (gitignored)
models/                   # checkpoints + labels.json (gitignored)
notebooks/flower_recognition.ipynb   # standalone Colab notebook
reports/                  # evaluation outputs
src/
  api.py                  # FastAPI service
  config.py               # YAML loader
  data.py                 # split + tf.data pipeline + augmentation
  evaluate.py             # test metrics + error analysis
  models.py               # baseline CNN + MobileNetV2 transfer
  predict.py              # FlowerPredictor (used by CLI and API)
  train.py                # training script
  utils.py                # logging, seeding, json helpers
```

## Try the inference (3 steps)

Sample images are included in `samples/` so you don't need the full dataset.

**1. Start the API**
```bash
uvicorn src.api:app --port 8000
```

**2. Open the interactive docs**
```
http://localhost:8000/docs
```
Click **POST /predict → Try it out → Choose File** → pick any image from `samples/` → **Execute**.

**3. Or use the CLI**
```bash
python -m src.predict samples/rose.jpg
python -m src.predict samples/daisy.jpg --top-k 3
```

---

## Training (if you want to retrain from scratch)

Requires the full dataset under `data/flowers/<class>/*.jpg`. Easiest to do on Colab — see the notebook section below.

```bash
python -m src.train --model transfer   # MobileNetV2 — strong model (~86% test acc)
python -m src.train --model baseline   # small CNN from scratch — comparison baseline
```

After training, evaluate on the held-out test split:

```bash
python -m src.evaluate
```

Outputs go to `reports/`: accuracy + macro F1, per-class P/R/F1, confusion matrix CSV + PNG, misclassified image grid.

## Running on Colab (GPU)

`notebooks/flower_recognition.ipynb` is fully self-contained — no repo clone needed. Just:

1. On your Google Drive, put the dataset under `MyDrive/data/flowers/` with one subfolder per class (`daisy/ dandelion/ rose/ sunflower/ tulip/`).
2. Upload the notebook to Colab.
3. Runtime → Change runtime type → Hardware accelerator: **GPU**.
4. Run all cells.

The first code cell mounts Drive and copies the dataset into local Colab storage (training off Drive directly is slow). The last cell zips up `models/` + `reports/` and triggers a browser download.

If your Drive folder is somewhere other than `MyDrive/data/flowers/`, edit `DRIVE_DATA_PATH` at the top of cell 2.

## Approach + decisions

**Why this dataset.** Brief said no MNIST/CIFAR-10. Flower-Recognition has variable resolution/aspect ratios (real preprocessing), visually similar pairs (rose↔tulip, daisy↔sunflower — meaningful for error analysis), mild class imbalance (~1.4×, so explicit handling makes sense), and is small enough to train on a laptop.

**Why two models.** A from-scratch CNN gives an honest lower bound. Without it, claiming "transfer learning works well here" is a bit hand-wavy — the baseline lets me put a number on the gain. Same train/val/test split, same augmentation, same class weights, same seed, so the comparison is fair.

**Why MobileNetV2.** Small (~3.5M params), trains in a few minutes on a T4, sub-100ms inference on CPU. Big enough that ImageNet features transfer well to flowers, small enough that I'm not pretending to need an A100. EfficientNetV2/ConvNeXt would probably score a bit higher but also overshoot the brief.

**Two-phase training.** Frozen backbone + head only first (lr=1e-3, 10 epochs), then unfreeze top 30 layers at lr=1e-5 for fine-tuning. BatchNorm layers stay frozen during fine-tune — small batch + small dataset = unstable BN running stats.

**Class imbalance.** Inverse-frequency class weights via Keras' `class_weight=`. Imbalance is mild so I didn't bother with oversampling — would just slow training without real gain.

**Augmentation.** Horizontal flip, ±20° rotation, ±15% zoom/shift, mild brightness. Skipped vertical flip (unnatural orientation) and aggressive cutout (close-ups would lose the subject).

**Image size.** Tried 128/160/224. 160 was the best speed/accuracy tradeoff on CPU; 224 was 2× slower for ~1pp.

## Failure modes seen

- **rose ↔ tulip**: closed buds look almost identical. Hard examples mining or a fine-grained second stage would help.
- **daisy ↔ sunflower**: yellow-centred radial petals in close-ups. Forcing more context (less zoom) might help.
- **Occluded / partially-cropped subjects**: model defaults to a frequent class. CutMix/Mosaic might help.
- A few obvious mislabels in the training set. Not worth manually cleaning at this scale.

## With more compute

- Multi-GPU `MirroredStrategy`, mixed precision, batch ≥256, cosine LR schedule.
- Optuna sweep over LR, dropout, augmentation strength, image size.
- Bigger backbone (EfficientNetV2-S / ConvNeXt-Tiny) and end-to-end fine-tune.
- Test-time augmentation at inference (free accuracy bump).
- Temperature scaling so the API's confidence is calibrated.
- Active-learning loop targeting the rose↔tulip confusions.
- ONNX/TFLite export for ~3–5× CPU inference speedup; Prometheus `/metrics` for drift.

## Reproducibility

Single seed in `configs/default.yaml`, set globally for python/numpy/TF. Splits are deterministic for a given seed. Hyperparameters all live in the config file.
