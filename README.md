# Khmer OCR Pipeline

A CNN + BiLSTM OCR system for Khmer text line recognition, built with PyTorch and CTC loss.

---

## Architecture

```
Image (grayscale, H=32, variable W)
       │
       ▼
┌─────────────┐
│  CNN        │  4 conv blocks → extracts local glyph features
│  Backbone   │  (subscript consonants, stacked vowels, diacritics)
└──────┬──────┘
       │ (T, B, 256)  — W' time steps, each a 256-dim feature
       ▼
┌─────────────┐
│  BiLSTM     │  2 layers, 256 units/direction
│             │  reads left→right AND right→left simultaneously
└──────┬──────┘
       │ (T, B, 512)
       ▼
┌─────────────┐
│  Linear     │  projects to vocabulary size
└──────┬──────┘
       │ (T, B, num_classes)
       ▼
   CTC Decode → Khmer text string
```

- **CNN** captures local pixel patterns (stacked glyphs, subscript consonants, diacritics)
- **BiLSTM** reads left→right and right→left so every position has full context
- **CTC loss** requires no character-level alignment — just image + full label string

---

## Project Structure

```
khmer_ocr/
├── data/
│   ├── train.parquet         ← your training data goes here
│   ├── val.parquet           ← your validation/test data goes here
│   └── dataset.py            # parquet loader, image transforms, batch collation
├── models/
│   └── crnn.py               # CNN + BiLSTM model + CTC loss wrapper
├── outputs/                  # checkpoints saved here after training
├── utils/
│   └── vocab.py              # Khmer character set, encode/decode, CTC decode
├── train.py                  # train → validate → final eval, all in one
├── interactive_predict.py    # load model once, predict images interactively
├── evaluate.py               # standalone evaluation on val.parquet
├── predict.py                # inference on images or parquet
└── requirements.txt
```

---

## What To Do Next (Step by Step)

### Step 1 — Install dependencies

```bash
cd khmer_ocr
pip install -r requirements.txt
```

### Step 2 — Place your data files

Copy your parquet files into the `data/` folder:

```
khmer_ocr/
└── data/
    ├── train.parquet
    └── val.parquet
```

Your parquet files need exactly two columns:
- `image` — image stored as bytes (HuggingFace format, raw bytes, or PIL Image)
- `text`  — the ground-truth Khmer string for that image line

### Step 3 — Check your vocab covers your data

Open `utils/vocab.py` and look at `KHMER_CHARS`. If your dataset contains characters not in that string (punctuation, Latin letters, special symbols), add them there before training. The dataset loader will warn you and drop any rows with unknown characters.

To quickly inspect what characters are in your data, run this one-liner:

```python
import pandas as pd
df = pd.read_parquet("data/train.parquet")
chars = set("".join(df["text"].astype(str).tolist()))
print(sorted(chars))
```

### Step 4 — Run a quick sanity check

Before committing to a full training run, verify the model and data load correctly:

```python
# Run from the khmer_ocr/ directory
python -c "
from utils.vocab import CHAR2IDX, NUM_CLASSES
from data.dataset import build_dataloader
from models.crnn import KhmerOCR
import torch

loader = build_dataloader('data/train.parquet', CHAR2IDX, batch_size=4)
images, labels, label_lengths, texts = next(iter(loader))
print('Image batch shape:', images.shape)
print('Sample text:', texts[0])

model = KhmerOCR(num_classes=NUM_CLASSES)
out = model(images)
print('Model output shape:', out.shape)
print('All good — ready to train!')
"
```

If this prints without errors you are ready.

### Step 5 — Train

```bash
python train.py \
    --train_parquet data/train.parquet \
    --val_parquet   data/val.parquet \
    --epochs        50 \
    --batch_size    32 \
    --output_dir    outputs/
```

Training will print a log like this every epoch:

```
 Epoch | Train Loss | Train CER |  Val Loss |  Val CER
------------------------------------------------------------
     1 |     3.2145 |    0.9821 |    3.1034 |   0.9654
     5 |     1.8432 |    0.5210 |    1.9012 |   0.5401
    20 |     0.4231 |    0.1203 |    0.5102 |   0.1450  ← best
   ...
```

At the end it automatically runs a final evaluation on `val.parquet` using the best checkpoint and prints CER, WER, and sample predictions. The best model is saved to `outputs/best_model.pth`.

Key training flags:

| Flag | Default | When to change |
|------|---------|----------------|
| `--epochs` | 50 | Increase to 100 if CER is still improving at epoch 50 |
| `--batch_size` | 32 | Lower to 16 if you run out of GPU/CPU memory |
| `--lr` | 1e-3 | Usually fine; reduces automatically on plateau |
| `--rnn_hidden` | 256 | Raise to 512 for more capacity, lower to 128 for speed |
| `--rnn_layers` | 2 | 2 is usually enough; try 3 if CER plateaus early |
| `--img_height` | 32 | Match your image heights; 32 works for most line images |

### Step 6 — Evaluate separately (optional)

If you want to re-run evaluation on the val set without retraining:

```bash
python evaluate.py \
    --checkpoint outputs/best_model.pth \
    --val_parquet data/val.parquet
```

### Step 7 — Predict on new images

**The model is saved automatically.** After training finishes, `outputs/best_model.pth` contains everything — weights, settings, epoch. You never need to retrain just to run predictions.

**Option A — Interactive mode (recommended for manual testing)**

Run once, then keep typing image paths until you are done:

```bash
python interactive_predict.py
# uses outputs/best_model.pth by default

python interactive_predict.py --checkpoint outputs/best_model.pth
```

It will prompt you like this:

```
=======================================================
  Khmer OCR — Interactive Prediction
=======================================================
  Loading checkpoint: outputs/best_model.pth
  Device     : cuda
  Image height: 32px
=======================================================

Image path: /path/to/my_line.png
Prediction : នេះជាការសាកល្បង

Image path: another_line.jpg
Prediction : សួស្តីប្រទេសកម្ពុជា

Image path: quit
Exiting.
```

You can also drag and drop an image file into the terminal — the path pastes automatically.

**Option B — Single image via command line**

```bash
python predict.py --checkpoint outputs/best_model.pth --image my_line.png
```

**Option C — Batch predict a folder or parquet**

```bash
python predict.py --checkpoint outputs/best_model.pth --image_dir my_images/
python predict.py --checkpoint outputs/best_model.pth --parquet new_data.parquet
```

---

## Troubleshooting

**"Dropped N rows with out-of-vocab characters"** — some text labels contain characters not in `KHMER_CHARS`. Add them to `vocab.py` or accept that those rows are skipped.

**Loss is NaN from the start** — usually means label lengths are longer than the model's output sequence. Try increasing `--img_height` or making sure images are not too narrow.

**Out of memory** — reduce `--batch_size` to 16 or 8.

**CER stuck above 50% after 20 epochs** — check that your image column is loading correctly (run the sanity check in Step 4) and that text labels are clean.

---

## Expected Performance

| Epoch | Typical CER |
|-------|------------|
| 5     | ~40–60%    |
| 20    | ~10–25%    |
| 50    | ~3–10%     |

Results depend heavily on dataset size and label quality. With 10k+ clean line images you can expect CER below 5% at epoch 50.

---

## Extending the Model

**More characters** — edit `KHMER_CHARS` in `utils/vocab.py`.

**More capacity** — raise `--rnn_hidden 512` or `--rnn_layers 3`.

**Faster / lighter** — lower `--rnn_hidden 128`, `--rnn_layers 1`.

**Stronger augmentation** — edit `get_transforms()` in `data/dataset.py`.