Intelligent Document ID Processing System
A local, privacy-first system to classify and extract key fields from Indian government-issued identity documents (PAN Card, Aadhaar Card) using deep learning–based OCR and spatial heuristics.

No data leaves your machine. All processing runs entirely locally — no external APIs, no cloud services.

✨ Features
Capability	Details
Document Classification	CNN-based (MobileNetV2 backbone) with keyword/regex heuristic fallback
Field Extraction	Deep learning OCR (PaddleOCR / EasyOCR) + label-proximity spatial analysis
Supported Documents	PAN Card, Aadhaar Card (extensible to Passport, Driving License)
Extracted Fields	Name, DOB, PAN Number, Aadhaar Number, Father's Name, Gender, Address
Post-Processing	Regex validation (PAN XXXXX0000X, Aadhaar 0000-0000-0000), date normalisation to YYYY-MM-DD, per-field confidence scoring
Preprocessing	Auto-deskew, denoising, CLAHE contrast enhancement, unsharp-mask sharpening
Input Formats	JPEG, PNG, PDF (first page extracted via PyMuPDF)
Robustness	Handles minor blur, skew, lighting variation, and image noise
UI	Interactive Streamlit dashboard with annotated image preview, confidence bars, raw OCR table, downloadable JSON
🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐
│ Input (Image / PDF) │
└─────────────────────┬───────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Preprocessing Pipeline │
│ Resize → Deskew → Denoise → CLAHE → Sharpen │
└─────────────────────┬───────────────────────────────────────┘
│
┌───────────┴───────────┐
▼ ▼
┌──────────────────┐ ┌──────────────────────┐
│ OCR Engine │ │ Full Text Output │
│ (PaddleOCR / │──▶│ (for classifier) │
│ EasyOCR) │ └──────────┬───────────┘
└────────┬─────────┘ │
│ ▼
│ ┌──────────────────┐
│ │ Document │
│ │ Classifier │
│ │ (CNN / Heuristic) │
│ └────────┬─────────┘
│ │
│ ▼ document_type
│ ┌──────────────────┐
└─────────────▶│ Field Extractor │
│ (OCR regions + │
│ spatial rules + │
│ regex matching) │
└────────┬─────────┘
│
▼
┌──────────────────┐
│ Post-Processing │
│ • Regex validate │
│ • Date normalise │
│ • Confidence │
└────────┬─────────┘
│
▼
┌──────────────────┐
│ JSON Output │
│ + UI Render │
└──────────────────┘
Upload a document
Drag and drop a PAN or Aadhaar card image (JPEG/PNG) or PDF. Results appear in four tabs:

Tab
Contents
📋 Extracted Fields	Key fields with confidence bars
🖼️ Image View	Original vs annotated with bounding boxes
🔍 Raw OCR	Every text region the OCR engine detected
{ } JSON Output	Structured JSON with download button

📁 Project Structure
text

id-document-processor/
├── app.py                        # Streamlit web application (main entry point)
├── config.py                     # Centralised configuration & constants
├── requirements.txt              # Python dependencies
├── train_classifier.py           # CNN classifier training script
├── README.md                     # This file
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          # Image preprocessing pipeline
│   │                             #   - Resize, deskew, denoise, CLAHE, sharpen
│   │                             #   - PDF → image conversion (PyMuPDF)
│   ├── postprocessing.py         # Validation & normalisation
│   │                             #   - PAN format validation (XXXXX0000X)
│   │                             #   - Aadhaar format validation (12 digits)
│   │                             #   - DOB normalisation (→ YYYY-MM-DD)
│   │                             #   - Name cleanup, confidence scoring
│   └── ocr_engine.py             # OCR adapter (PaddleOCR / EasyOCR)
│                                 #   - Unified interface, auto-detection
│                                 #   - Compatible with v2/v3/v4 APIs
│
└── models/
    ├── __init__.py
    ├── classifier.py             # Document type classifier
    │                             #   - CNN (MobileNetV2) if weights exist
    │                             #   - Heuristic fallback (keywords + regex)
    └── extractor.py              # Key field extraction engine
                                  #   - Label-proximity spatial analysis
                                  #   - Regex-based field identification
                                  #   - Address multi-line merging
🧠 Model Details
Document Classifier
Aspect
Detail
Architecture	MobileNetV2 (pretrained on ImageNet) + custom classification head
Input	224×224×3 RGB image
Output	Softmax over 3 classes: Aadhaar, PAN, Unknown
Training	Optional — train_classifier.py provided
Fallback	Keyword + regex heuristic (works without training)

The heuristic classifier achieves high accuracy by leveraging:

PAN-specific signals: "INCOME TAX DEPARTMENT", "Permanent Account Number", pattern [A-Z]{5}[0-9]{4}[A-Z]
Aadhaar-specific signals: "AADHAAR", "UIDAI", "Unique Identification", pattern \d{4}[\s-]?\d{4}[\s-]?\d{4}
OCR Engine
Engine
Model
Type
PaddleOCR	PP-OCRv4	CRNN + CTC (detection: DBNet, recognition: SVTR)
EasyOCR	English v1	CRNN + CTC (detection: CRAFT, recognition: ResNet+LSTM)

Both are deep learning–based end-to-end OCR systems — not rule-based or template matching.

Field Extractor
Uses a multi-strategy approach:

Direct regex match — for PAN number and Aadhaar number (unique formats)
Label-proximity — find "Name:", "DOB:" labels, then grab the nearest text region to the right/below
Spatial heuristics — address is in the bottom 40% of Aadhaar cards; gender is a short keyword
Multi-line merging — address lines are combined based on vertical proximity
🏋️ Training Your Own Classifier (Optional)
If the heuristic classifier isn't sufficient (e.g., adding Passport or Driving License support), train a CNN:

1. Organise training data
text

training_data/
├── Aadhaar/        # 50+ images
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
├── PAN/            # 50+ images
│   ├── img001.jpg
│   └── ...
└── Unknown/        # optional — non-ID images
    └── ...
2. Train
bash

python train_classifier.py --data_dir ./training_data --epochs 30
3. Result
The trained model is saved to models/weights/document_classifier.keras and loaded automatically by the app. The CNN is tried first; if confidence is below 70%, it falls back to the heuristic.

⚙️ Configuration
All tunable parameters are in config.py:

Parameter
Default
Description
PREPROCESSING.max_dimension	2000	Max pixel dimension for resize
PREPROCESSING.denoise_strength	10	OpenCV fastNlMeans strength
PREPROCESSING.clahe_clip_limit	2.0	CLAHE contrast clip limit
PREPROCESSING.sharpen_strength	0.8	Unsharp mask strength
CLASSIFICATION.unknown_threshold	0.30	Below this → classified as "Unknown"

These can also be adjusted live via the Streamlit sidebar.

📊 Evaluation Criteria Mapping
Requirement Area
Implementation
Weight
Document Classification	CNN (MobileNetV2) + keyword/regex heuristic	30%
Model-Based Field Extraction	PaddleOCR/EasyOCR (DL-based) + spatial analysis	55%
Post-Processing & Clarity	Regex validation, date normalisation, confidence scoring	15%

⚠️ Assumptions & Limitations
Aspect
Detail
Image quality	Works best with clear, well-lit photos or scans. Extremely blurry or dark images may fail.
Language	English text only. Hindi/regional text on Aadhaar is ignored.
Handwriting	Not supported — printed text only.
Layout variants	Assumes standard government-issued card layouts. Non-standard formats may misextract.
Address accuracy	Multi-line address merging is heuristic-based; may include/exclude adjacent text.
Classification	Heuristic fallback relies on OCR quality. If OCR fails entirely, classification falls to "Unknown".
Single document	Processes one document per upload. Multiple cards in one image are not handled.
PDF	Only the first page is processed.

🛠️ Tech Stack
Component
Technology
Frontend UI	Streamlit 1.28+
OCR (primary)	PaddleOCR (PP-OCRv4)
OCR (fallback)	EasyOCR
Image Processing	OpenCV 4.8+, Pillow
Classification	TensorFlow/Keras (MobileNetV2)
PDF Support	PyMuPDF
Language	Python 3.9 – 3.12