# Document OCR: Full-Page Text Recognition

This extension adds full-document OCR capabilities to the Khmer OCR project using line segmentation. Instead of recognizing only single lines of text, you can now process entire document pages.

## Architecture Overview

```
Document Image
       │
       ▼
┌──────────────────────────────────────┐
│  Line Segmentation Module            │
│  ├─ Grayscale conversion             │
│  ├─ Binary thresholding              │
│  ├─ Horizontal projection profile    │
│  └─ Line boundary detection          │
└──────────────────┬───────────────────┘
                   │ (Line images)
                   ▼
        ┌──────────────────────┐
        │  Line Recognition    │
        │  (for each line)     │
        │                      │
        │  ┌────────────────┐  │
        │  │ CNN Backbone   │  │
        │  ├────────────────┤  │
        │  │ BiLSTM         │  │
        │  ├────────────────┤  │
        │  │ CTC Decoder    │  │
        │  └────────────────┘  │
        └──────────────────────┘
                   │
                   ▼
        Full Document Text
        (lines separated by \n)
```

## Line Segmentation Algorithm

The line segmentation uses **horizontal projection profiles** to detect text lines:

### Step 1: Grayscale Conversion
Convert the input image to grayscale (single channel).

### Step 2: Binary Threshold
Apply binary thresholding to create a binary image (0 = background, 255 = text):
```
threshold = 127  # adjustable parameter
binary = (grayscale < threshold) ? 255 : 0
```

### Step 3: Horizontal Projection Profile
Sum the black pixels in each row to create a 1D projection profile:
```
projection[row] = sum(binary[row, :]) / 255
```

This shows the "density" of text pixels in each row. Rows with no text have projection=0.

### Step 4: Line Boundary Detection
Find consecutive groups of non-zero projection rows:
- Gaps larger than `min_gap` pixels separate different lines
- Groups with height ≥ `min_height` pixels are considered valid lines

## Usage

### Basic Document Processing

```bash
python document_predict.py \
    --checkpoint outputs/best_model.pth \
    --image path/to/document.png \
    --output result.txt
```

**Output:**
```
[1] Segmenting document into lines...
    ✓ Found 5 lines
    Projection stats: 120 text rows, max=45

[2] Recognizing text in each line...
    Line 1 [rows 0-23, h=24]: បង្រាយលក្ខណ៍គឺលេខ
    Line 2 [rows 27-47, h=21]: ភាសាខ្មែរគឺស្នេហ៍
    Line 3 [rows 50-72, h=23]: វប្បធម៌នៃខ្មែរ
    Line 4 [rows 75-96, h=22]: រズវាងមនុស្ស
    Line 5 [rows 100-119, h=20]: ដែលបង្កើតពីស្នេហ៍

[3] Combined 5 lines into document

========================================================================
DOCUMENT TEXT:
========================================================================
បង្រាយលក្ខណ៍គឺលេខ
ភាសាខ្មែរគឺស្នេហ៍
វប្បធម៌នៃខ្មែរ
រズវាងមនុស្ស
ដែលបង្កើតពីស្នេហ៍
========================================================================
    ✓ Saved to result.txt
```

### Using Updated predict.py

The original `predict.py` now supports a `--document` flag:

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --document path/to/document.png
```

### Testing Line Segmentation

Use the test script to visualize how line segmentation works:

```bash
python test_line_segmentation.py \
    --image path/to/document.png \
    --threshold 127 \
    --min_gap 3 \
    --min_height 5
```

This shows:
- Grayscale conversion stats
- Binary thumbnail stats
- Horizontal projection profile (first 50 rows)
- Detected line boundaries with heights

Example output:
```
[1] Converting to grayscale...
    ✓ Shape: (600, 800), dtype: uint8
    Min: 0, Max: 255, Mean: 200.4

[2] Applying binary threshold (threshold=127)...
    ✓ Black pixels: 24500 (5.1% coverage)

[3] Computing horizontal projection profile...
    ✓ Total rows: 600
    ✓ Text rows: 120
    ✓ Max projection: 45
    ✓ Mean projection: 2.35

[4] Detecting lines (min_gap=3, min_height=5)...
    ✓ Detected 5 lines
       Line 1: rows 0-23 (height=24)
       Line 2: rows 27-47 (height=21)
       Line 3: rows 50-72 (height=23)
       Line 4: rows 75-96 (height=22)
       Line 5: rows 100-119 (height=20)

[5] Projection profile (first 50 rows):
    Row   0: ███████████████████░░░░░░░ [IN LINE]
    Row   1: ████████████████████░░░░░░ [IN LINE]
    ...
```

## Parameter Tuning

### Threshold (default: 127)
Controls binary thresholding. Lower values → more text detected, higher values → less text.
- Problem: Too few lines? Try lower threshold
- Problem: Too many lines? Try higher threshold
- Range: 0-255

### min_gap (default: 3)
Minimum pixel gap between lines. Larger values merge closer lines.
- Problem: Merging lines together? Increase min_gap
- Problem: Splitting single lines? Decrease min_gap
- Typical range: 2-10

### min_height (default: 5)
Minimum line height in pixels. Filters out noise and small artifacts.
- Problem: Detecting noise as lines? Increase min_height
- Problem: Missing short lines? Decrease min_height
- Typical range: 3-20

### expand (default: 2)
Margins to add around detected lines (top/bottom pixels).
- Provides context for the model
- Typical range: 0-5

## Advanced: Programmatic Usage

### In Python Code

```python
from utils.line_segmentation import segment_document
from PIL import Image
import torch
from models.crnn import KhmerOCR
from predict import predict_image

# Load document
doc = Image.open("document.png")

# Segment into lines
line_images, line_bounds = segment_document(
    doc,
    threshold=127,
    min_gap=3,
    min_height=5,
    expand_margin=2,
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KhmerOCR(num_classes=200).to(device)
model.load_state_dict(torch.load("outputs/best_model.pth")["model_state_dict"])
model.eval()

# Process each line
results = []
for line_img in line_images:
    text = predict_image(model, line_img, img_height=32, device=device)
    results.append(text)

# Combine
full_text = "\n".join(results)
```

## Line Segmentation Module API

**File:** `utils/line_segmentation.py`

### Functions

#### `segment_document(img, threshold=127, min_gap=3, min_height=5, expand_margin=0)`
Main function for line segmentation.
- **img**: PIL Image of the document
- **threshold**: Binary threshold (0-255)
- **min_gap**: Min gap between lines (pixels)
- **min_height**: Min line height (pixels)
- **expand_margin**: Margin to expand around lines (pixels)
- **Returns**: `(line_images, line_bounds)` tuple

#### `image_to_array(img)`
Convert PIL Image to grayscale numpy array.

#### `apply_binary_threshold(gray, threshold=127)`
Apply binary thresholding.

#### `compute_horizontal_projection(binary)`
Compute horizontal projection profile.

#### `detect_line_boundaries(projection, min_gap=3, min_height=5)`
Detect line boundaries from projection.
- **Returns**: List of `(start_row, end_row)` tuples

#### `get_line_stats(projection)`
Get statistics about projection profile.

## Integration with Existing Pipeline

### Before (Single Line)
```
document image → [manual line extraction] → 
single line image → predict.py → text
```

### After (Full Document)
```
document image → document_predict.py → 
  [line segmentation] → multiple line images → 
  [batch recognition] → full_text
```

The existing CRNN model is unchanged. Line segmentation is a preprocessing step that extracts lines for the model to process.

## Performance Notes

- **Speed**: Depends on document size and number of lines
  - 500x600 document: ~0.5-1 second (excluding model inference)
  - Model inference: ~20-50ms per line
- **Memory**: Loads full document in memory
  - Typical documents: <100MB

## Troubleshooting

### "No lines detected"
- Image may be too small or have poor contrast
- Try lowering `--threshold` or increasing `--min_height`
- Check if image is actually a document (not already single lines)

### "Too many lines detected"
- Likely detecting noise as lines
- Try increasing `--min_height` or `--min_gap`
- Or increase `--threshold` (less aggressive binary conversion)

### "Lines are merged together"
- Reduce text density or increase `--min_gap`
- Document may have very close line spacing

### "Lines are split in the middle"
- Vertical text strokes detected as line boundaries
- Decrease `--min_gap`

## Files Added/Modified

**New files:**
- `utils/line_segmentation.py` - Line segmentation module
- `document_predict.py` - Full document OCR script
- `test_line_segmentation.py` - Testing/visualization script

**Modified files:**
- `predict.py` - Added `--document` flag and `predict_document()` function
- `requirements.txt` - Added `opencv-python>=4.8.0`

## Future Improvements

Possible enhancements:
1. **Automatic threshold selection** using Otsu's algorithm
2. **Multi-column document detection** for newspaper/book pages
3. **Skew correction** for rotated pages
4. **Line confidence scores** to filter uncertain lines
5. **Batch processing** for multiple documents
6. **GPU-accelerated segmentation** for faster processing
