# 🚀 Quick Start: Document OCR

## What to Run

### 1. **Interactive Mode** (Recommended for testing) ⭐
```bash
python interactive_document_predict.py
```

Copy-paste document image paths one at a time. Results show line-by-line with full document at the end.

**Example session:**
```
Khmer OCR — Interactive Full Document Prediction
========================================================================
  Loading checkpoint: outputs/best_model.pth
  Device           : cpu
  Image height     : 32px
  Threshold        : 127
  Min gap          : 3px
  Min line height  : 5px
  Expand margin    : 2px
========================================================================
  Enter a document image path to predict.
  You can copy-paste the full path from your file manager.
  Type 'quit' or press Ctrl+C to exit.
========================================================================

Document path: C:\Users\mariya\Desktop\OCR\testing_image\page1.png

  [800x600] RGB image
  Detected 5 lines
    Line  1 [rows    0-  23, h=24]: បង្រាយលក្ខណ៍គឺលេខ
    Line  2 [rows   27-  47, h=21]: ភាសាខ្មែរគឺស្នេហ៍
    Line  3 [rows   50-  72, h=23]: វប្បធម៌នៃខ្មែរ
    Line  4 [rows   75-  96, h=22]: រズវាងមនុស្ស
    Line  5 [rows  100- 119, h=20]: ដែលបង្កើតពីស្នេហ៍

────────────────────────────────────────────────────────────────────────
FULL DOCUMENT TEXT (5 lines)
────────────────────────────────────────────────────────────────────────
បង្រាយលក្ខណ៍គឺលេខ
ភាសាខ្មែរគឺស្នេហ៍
វប្បធម៌នៃខ្មែរ
រズវាងមនុស្ស
ដែលបង្កើតពីស្នេហ៍
────────────────────────────────────────────────────────────────────────

Document path: quit
Exiting.
```

### 2. **Batch Processing** (Process one document, save to file)
```bash
python document_predict.py \
    --checkpoint outputs/best_model.pth \
    --image C:\path\to\document.png \
    --output result.txt
```

### 3. **Command-line Mode** (Using predict.py with document flag)
```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --document C:\path\to\document.png
```

### 4. **Test Line Segmentation** (Debug/visualize)
```bash
python test_line_segmentation.py \
    --image C:\path\to\document.png \
    --tune
```

---

## Parameter Tuning

If line detection isn't working well, try adjusting:

### Too few lines detected?
```bash
python interactive_document_predict.py \
    --threshold 100 \
    --min_gap 2 \
    --min_height 3
```

### Too many lines detected (detecting noise)?
```bash
python interactive_document_predict.py \
    --threshold 150 \
    --min_gap 5 \
    --min_height 10
```

### Lines are splitting in the middle?
```bash
python interactive_document_predict.py --min_gap 2
```

### Lines are merging together?
```bash
python interactive_document_predict.py --min_gap 8
```

---

## Reference: All Parameters

| Script | Parameter | Default | Range | Purpose |
|--------|-----------|---------|-------|---------|
| `interactive_document_predict.py` | `--threshold` | 127 | 0-255 | Binary threshold for line detection |
| | `--min_gap` | 3 | 1-20 | Min gap (px) between lines |
| | `--min_height` | 5 | 1-50 | Min line height (px) |
| | `--expand` | 2 | 0-10 | Margin around lines (px) |
| | `--quiet` | - | flag | Suppress line-by-line output |
| | `--checkpoint` | `outputs/best_model.pth` | - | Model checkpoint path |
| `document_predict.py` | (all same as above) | | | |
| `test_line_segmentation.py` | (all same as threshold, min_gap, min_height) | | | |

---

## Comparison: Single Line vs Full Document

### Single Line Recognition
```bash
# Only for single line images
python interactive_predict.py
Document path: C:\single_line.png
Prediction: បង្រាយលក្ខណ៍គឺលេខ
```

### Full Document Recognition  
```bash
# For multi-line document pages
python interactive_document_predict.py
Document path: C:\full_page.png
Detected 5 lines
    Line 1: បង្រាយលក្ខណ៍គឺលេខ
    Line 2: ភាសាខ្មែរគឺស្នេហ៍
    ...
```

---

## File Locations

Copy paths from Windows File Manager:
1. Open File Manager
2. Navigate to your document image
3. Hold **Shift** and right-click
4. Select **"Copy as path"**
5. Paste into the terminal

Example: `C:\Users\mariya\Desktop\OCR\testing_image\document.png`

---

## Troubleshooting

### "Model not found"
```bash
# Make sure you have trained the model first
python train.py --train_parquet data/train.parquet --val_parquet data/val.parquet
```

### "No lines detected"
- Document might be too small or very light
- Try lowering threshold: `--threshold 100`
- Check image isn't already a single line

### "Takes too long"
- Working on GPU? Check `Device: cuda` is shown
- Document very large? Split into smaller pages first

### Segmentation looks off?
- Test with: `python test_line_segmentation.py --image path --tune`
- Adjust parameters based on visualized projection profile

---

## Keyboard Shortcuts

| In interactive mode |  |
|---------------------|--|
| Ctrl+C | Exit program |
| Type `quit` or `exit` | Exit program |
| Paste full path | Just paste—handles quotes automatically |

---

## Next Steps

1. ✅ Try: `python interactive_document_predict.py`
2. ✅ Paste a document path
3. ✅ Review line-by-line results
4. ✅ Adjust parameters if needed
5. ✅ Save output for batch processing
