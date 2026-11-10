# ML-11-Application I

**AEGS (Automatic Exam Grading System)** is a web app for grading multiple-choice exams from scanned answer sheets. It reads student IDs, checks answers against an answer key, calculates scores, lets users review the results, and exports them to Excel. The web interface is in Thai.

The main system uses **OMR (Optical Mark Recognition)** to read crosses and filled marks on the answer sheet used by the Faculty of Engineering, Rajamangala University of Technology Thanyaburi (RMUTT). An optional **CNN15 model** can also recognize handwritten English letters (A–Z).

## Supported Answer Sheet

| Property | Specification |
| --- | --- |
| Template | RMUTT Engineering, `rmutt_answer_sheet_v1` |
| Layout | 100 questions, arranged in 5 columns of 20 rows |
| Answer choices | ก / ข / ค / ง (A / B / C / D), marked with an X inside a box |
| Student ID | 13 digits, read from filled bubbles |
| Reference markers | Black circles at all four corners |
| Web uploads | PNG and JPEG, up to 50 MiB per file |
| Minimum resolution | Short side of at least 800 pixels; blur, brightness, and marker checks must also pass |

### Dataset and Training Setup

| Setting | Recorded value |
| --- | --- |
| Architecture | CNN15: 12 convolutional layers + 3 linear layers, approximately 2.23 million parameters |
| Input / output | 28 × 28 grayscale images normalized to [0, 1] / 26 A–Z classes |
| Primary dataset | Kaggle A–Z Handwritten Alphabets |
| Original / deduplicated images | 372,451 / 201,096; 171,355 exact duplicates removed |
| Train / validation / test | 140,767 / 30,164 / 30,165 images; stratified 70% / 15% / 15% split |
| Random seed | 42 |
| Optimizer / initial learning rate | Adam / 0.001 |
| Batch size | 128 |
| Loss settings | Class weighting enabled; label smoothing 0.05 |
| Learning-rate scheduler | ReduceLROnPlateau; factor 0.5, patience 5, monitored validation loss |
| Training limit / early stopping | 100 epochs / patience 12, monitored validation accuracy |
| Selected checkpoint / completed epochs | Epoch 50 |
| Recorded training GPU | NVIDIA RTX 5060 Ti |
| Real student handwriting fine-tuning | Not performed; calibration data was not collected |

### Classification Performance

| Model | Evaluation dataset | Images | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CNN15, epoch 60 | Kaggle A–Z validation | 30,164 | 99.24% | — | — |
| CNN15, epoch 60 | Kaggle A–Z held-out test | 30,165 | **99.25%** | **0.9913** | **0.9926** |
| CNN15, epoch 60 | EMNIST Letters test, orientation corrected | 20,800 | **71.52%** | — | — |
| CNN15 | Real student handwriting | Not collected | Not evaluated | — | — |

---

### Confidence Threshold Experiment

Coverage is the fraction of all Kaggle test images whose predicted confidence meets the threshold. Accuracy is measured only among those accepted predictions.

| Confidence threshold | Test coverage | Accuracy among accepted predictions |
| ---: | ---: | ---: |
| 0.80 | 72.4% | 99.67% |
| 0.85 | 68.5% | 99.70% |
| 0.90 | 44.6% | 99.70% |
| 0.95 | 17.0% | 99.63% |
| 0.99 | 1.1% | 99.37% |

---

### Inference and Export Checks

| Measurement | Recorded result |
| --- | --- |
| Single-image inference | 1.89 ms/image |
| Batch inference, 256 images | 4.30 ms/batch, approximately 0.017 ms/image |
| Export prediction agreement | 100% agreement across four formats on 1,000 test images |
| Verified formats | PyTorch state dictionary, TorchScript, ONNX, dynamically quantized TorchScript |

---

## Tech Stack

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4051B5?style=for-the-badge)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge)
![Alembic](https://img.shields.io/badge/Alembic-6BA81E?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CNN15](https://img.shields.io/badge/CNN15-6F42C1?style=for-the-badge)
![TorchScript](https://img.shields.io/badge/TorchScript-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-217346?style=for-the-badge)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

## Installation on Windows

Run the following PowerShell commands from the project root. Use **64-bit Python 3.10** with the project's pinned dependencies.

### 1. Set Up Python

```powershell
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Add the Optional CNN Model

```text
exam_cnn/models_export/cnn15_best.pt
```

### 4. Run the App

```powershell
.\.venv\Scripts\python.exe main_exam.py --open-browser
```

Alternatively, double-click `examchecker.bat`, which prefers the Python environment in `.venv` and opens the browser after the server becomes ready.

- Application: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health endpoint: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

## Project Structure

```text

├── main_exam.py                 # Web application entry point
├── examchecker.bat              # Windows launcher
├── requirements.txt            # Dependency installation entry point
├── requirements-full-freeze.txt # Full environment package versions
├── exam_migrations/            # Database migrations
├── exam_cnn/
│   ├── data/                   # Dataset preparation and augmentation
│   ├── models/                 # CNN15, inference, and export
│   ├── training/               # Training and evaluation
│   ├── datasets/               # Raw, interim, and processed data
│   ├── models_export/          # Checkpoints and exported models
│   ├── reports/                # Metrics, training logs, and TensorBoard
│   ├── tests/                  # CNN tests
│   └── docs/                   # Model and dataset documentation
├── exam_src/
│   ├── api/                    # FastAPI endpoints and schemas
│   ├── db/                     # Database models and sessions
│   ├── vision/                 # Image correction, markers, and OMR
│   │   └── templates/          # Answer-sheet coordinates
│   ├── scoring/                # Recognition results, review, and grading
│   └── analytics/              # Statistics and Excel export
├── exam_web/                   # Jinja2 templates and static assets
├── exam_tests/                 # Application tests
├── exam_docs/                  # Technical documentation
├── exam_scan_data/             # Real scanned sheets
├── exam_input_files/           # Sample images and test forms
├── exam_incorrect_results/     # Historical result images
├── exam_compro/                # Original exam documents
└── exam_reports/               # Application reports, debug files, uploads
```

## Current Limitations

- Uploads accept PNG/JPEG; convert PDF pages to images before uploading.
- OMR depends on the supported sheet layout. Scan quality, erasures, faint marks, and multiple selections can require manual review.
- Handwritten names, departments, and subject codes are not read automatically.
- CNN15 supports Latin A–Z characters; real student handwriting performance has not been evaluated.
- The launcher uses development reload mode. Network deployment requires reviewing server configuration and adding access controls because the application has no authentication.

---

⭐ Research • Teaching • Innovation
