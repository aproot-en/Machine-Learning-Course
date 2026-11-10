# ML-11-Application I

**AEGS (Automatic Exam Grading System)** is a web app for grading multiple-choice exams from scanned answer sheets. It reads student IDs, checks answers against an answer key, calculates scores, lets users review the results, and exports them to Excel. The web interface is in Thai.

The main system uses **OMR (Optical Mark Recognition)** to read crosses and filled marks on the answer sheet used by the Faculty of Engineering, Rajamangala University of Technology Thanyaburi (RMUTT). An optional **CNN15 model** can also recognize handwritten English letters (A–Z).

## Main Features

- Create, rename, and manage exams by academic year.
- Create answer keys separately from exams and reuse existing keys.
- Set correct answers, points per question, and canceled questions in the web interface.
- Upload multiple answer sheet images and track grading progress.
- Check image quality, find the four corner markers, and align images before reading them.
- Read 13-digit student IDs from marked boxes and detect answers ก / ข / ค / ง (A / B / C / D).
- Review unclear marks, check the original images, and correct student IDs.
- Calculate scores automatically after grading or saving an answer key, with an option to recalculate.
- View individual scores, score statistics, question difficulty, discrimination, and answer-choice distribution.
- Export results for one exam to Excel, or combine results from multiple exams into separate sheets in one file.
- Save grading job status in the database and resume unfinished jobs after restarting the app.

## Supported Answer Sheet

| Item | Details |
| --- | --- |
| Form | RMUTT Engineering, using the `rmutt_answer_sheet_v1` template |
| Questions | 100 questions, arranged in 5 columns × 20 rows |
| Choices | ก / ข / ค / ง (A / B / C / D), marked with a cross |
| Student ID | 13-digit ID marked in number boxes |
| Alignment markers | Black circles at all four corners |
| Upload formats | PNG and JPEG, up to 50 MiB per file |
| Minimum resolution | Shorter side at least 800 pixels; images must also pass blur, brightness, and marker checks |

The exam setup page lets you choose the number of questions and answer choices, but the current image reader uses fixed positions from the 100-question form above. Other forms need an updated template and detection testing first.

Form details: [Real Answer Sheet Template](exam_docs/answer_sheet_real_template.md)


## 🛠️ Tech Stack

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

Run the following commands in **PowerShell** from the project root folder.

### 1. Set Up Python

Download or clone this repository and open PowerShell in the project folder. Use **64-bit Python 3.10** to match the pinned dependencies.

```powershell
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` includes the web app and testing packages, and references `exam_cnn/requirements.txt` for machine learning packages. The ML requirements use the PyTorch CUDA 12.8 package source. `requirements-full-freeze.txt` lists all package versions from the checked environment.

The installation also includes packages for model training and export, so it may take some time and disk space. An NVIDIA GPU is optional for CNN tasks. Model inference falls back to the CPU when CUDA is not available.

Check that the main packages can be imported:

```powershell
.\.venv\Scripts\python.exe -c "import torch, cv2, numpy; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The project specifies NumPy `1.26.4` and PyTorch `2.11.0+cu128`. Seeing `CUDA: False` is fine if you are using the CPU.

> The ML code uses PyTorch, not TensorFlow. TensorBoard 2.20.0 is used for training logs and compatibility with ONNX's protobuf dependencies. See [Environment Notes](exam_docs/environment.md) for issues found in an earlier environment.

### 3. Add the Optional CNN Model

The OMR grading system works without a CNN checkpoint. To use the A–Z handwriting prediction API, place a compatible checkpoint at:

```text
exam_cnn/models_export/cnn15_best.pt
```

Model files and datasets are excluded by `.gitignore`, so they are not included when you clone the repository. If you do not have a checkpoint, you can train one using the steps below.

The web app can still start without this file. However, `/api/predict/single` will return HTTP `503`, and `/api/health` will show `model_loaded: false`.

### 4. Run the App

```powershell
.\.venv\Scripts\python.exe main_exam.py --open-browser
```

You can also double-click `examchecker.bat`. It looks for Python in `.venv` first and opens the browser when the server is ready.

- Web app: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

At startup, Alembic automatically creates or updates the `aegs.db` database structure. Press `Ctrl+C` in the server terminal to stop the app.

The current version is designed for local use. All sessions share the same data, there is no login system, and the launcher binds the server to `127.0.0.1`.

## How to Use

1. Open the Dashboard and create an exam. Enter its name, academic year, question count, and answer choices to match the answer sheet.
2. Enter an answer key or copy one from an existing exam or key set. Set points per question and canceled questions, then save.
3. Open the upload page, select PNG/JPEG answer sheet images, and start grading.
4. Wait for grading to finish. Review student IDs and any answers flagged for checking. Confirm or correct them using the scanned images.
5. Check the scores and recalculate after making changes if needed. If you uploaded sheets before creating an answer key, save the key to calculate the scores.
6. View the statistics and export to Excel. You can set score weights. To combine exams, choose the exams, sheet names, and score weights for each one.

A single-exam Excel export has four sheets: **Scores**, **Statistics Summary**, **Question Analysis**, and **CLO**. A combined export creates separate score sheets for the selected exams. CLO information must already be set up in the database for that part of the report to contain data.

## Project Structure

```text
.
├── main_exam.py                  # Web app entry point
├── examchecker.bat               # Windows launcher
├── config.yaml                   # Data, model, form, and database settings
├── requirements.txt             # Main dependency installer
├── requirements-full-freeze.txt # Full package version list
├── alembic.ini
├── exam_migrations/             # Database migrations
├── exam_cnn/                    # ML code, data, models, and reports
│   ├── data/                    # Data preparation, cleaning, augmentation
│   ├── models/                  # CNN15, inference, and export
│   ├── training/                # Model training and evaluation
│   ├── datasets/                # Raw, interim, and processed data
│   ├── models_export/           # Checkpoints, TorchScript, and ONNX
│   ├── reports/                 # Evaluation results and training logs
│   ├── tests/                   # CNN tests
│   └── docs/                    # Model and dataset documentation
├── exam_src/
│   ├── api/                     # FastAPI, schemas, and endpoints
│   ├── db/                      # SQLAlchemy models and sessions
│   ├── vision/                  # Image alignment, markers, and OMR
│   │   └── templates/           # Answer sheet coordinates
│   ├── scoring/                 # Detection results, review, and scoring
│   └── analytics/               # Statistics, question analysis, Excel export
├── exam_web/
│   ├── templates/               # Jinja2 pages
│   └── static/                  # CSS and images
├── exam_tests/                  # Automated tests
├── exam_docs/                   # Technical documentation
├── exam_scan_data/              # Real scanned data
├── exam_input_files/            # Sample images and test forms
├── exam_incorrect_results/      # Previous incorrect results for reference
├── exam_compro/                 # Original exam documents
└── exam_reports/                # Reports, debug output, and uploaded files
```

## Configuration and Stored Data

The main settings are in [config.yaml](config.yaml), including data paths, model paths, training settings, and minimum image resolution.

ML code and files are under [exam_cnn](exam_cnn/README.md). The web app loads the model through `exam_cnn.models.inference`. Training data paths use `paths.data_*`, exported models use `paths.models_export`, and ML reports use `paths.ml_reports`.

- SQLite data is stored in `aegs.db` by default. The current database code does not support PostgreSQL.
- Uploaded images are saved in `exam_reports/uploads/<exam_id>/`.
- Answer sheet coordinates are defined in [rmutt_answer_sheet_v1.json](exam_src/vision/templates/rmutt_answer_sheet_v1.json).
- Some mark-detection thresholds are in [answer_detection.py](exam_src/vision/answer_detection.py), and image quality checks are in [preprocess.py](exam_src/vision/preprocess.py).
- Some model and inference settings are hardcoded, so check the related Python files when changing `config.yaml`.

To back up your work, stop the app and copy both `aegs.db` and `exam_reports/uploads/`. Deleting an exam in the app removes its related database records but does not delete uploaded images from disk.

Top-level project folders use the `exam_` prefix, while inner folder names stay the same. When starting with a database from the older folder structure, Alembic updates saved image and grading-job paths to match the new names.

## Training and Exporting the A–Z Model (Optional)

CNN15 takes **28 × 28 grayscale images** and predicts one of **26 English letters (A–Z)**. This is separate from the OMR answer-reading process.

Prepare the Kaggle A–Z dataset as described in [Dataset Provenance](exam_cnn/docs/dataset_provenance.md), then place the CSV file at:

```text
exam_cnn/datasets/raw/kaggle_az/A_Z Handwritten Data.csv
```

Run these commands in order from the project root:

```powershell
.\.venv\Scripts\python.exe -m exam_cnn.data.prepare_dataset
.\.venv\Scripts\python.exe -m exam_cnn.training.train
.\.venv\Scripts\python.exe -m exam_cnn.training.evaluate
.\.venv\Scripts\python.exe -m exam_cnn.models.export
```

Data preparation creates `exam_cnn/datasets/processed/az_dataset.npz`. Training saves `exam_cnn/models_export/cnn15_best.pt`. Export creates TorchScript, ONNX, and quantized TorchScript versions. Cross-dataset evaluation also needs EMNIST data prepared at the path described in the documentation.

The results recorded in the [Model Card](exam_cnn/docs/model_card.md) are **99.25% accuracy on the Kaggle test set** and **71.52% on EMNIST Letters**. These are results from earlier letter-classification experiments, **not the accuracy of the exam grading system**. The model has not yet been validated on real student handwriting.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover scoring, question analysis, Excel export, answer-key reuse, exam management, synthetic image reading, detection robustness, and background grading jobs. Some inference tests need a prepared checkpoint and dataset.

`pytest.ini` automatically includes tests from both `exam_tests` and `exam_cnn/tests`.

## Current Limitations

- The upload page accepts PNG/JPEG but not PDF directly. Convert PDF pages to images before uploading.
- OMR is designed for the supported form. Scan quality, erased marks, faint marks, or multiple marks may need manual review.
- The system does not automatically read handwritten names, departments, or course codes.
- The CNN model recognizes English letters A–Z; it is not used to classify Thai answer choices on the current form.
- The launcher uses development mode with auto-reload, and the app has no authentication. Review the server settings and add access controls before making it available over a network.

## More Documentation

- [Real Answer Sheet Template](exam_docs/answer_sheet_real_template.md)
- [Environment and Dependency Notes](exam_docs/environment.md)
- [Dataset Provenance](exam_cnn/docs/dataset_provenance.md)
- [Data Preparation](exam_cnn/docs/phase2_preprocessing.md)
- [CNN15 Model Design](exam_cnn/docs/phase3_model_design.md)
- [Model Evaluation](exam_cnn/docs/phase5_evaluation.md)
- [Model Card](exam_cnn/docs/model_card.md)
- [Image Processing Pipeline](exam_docs/phase7_9_vision_pipeline.md)
- [Scoring](exam_docs/phase10_11_scoring.md)
- [Web App and Testing](exam_docs/phase12_16_web_and_testing.md)

Some files in `exam_docs/` are notes from earlier development stages. For the current behavior, check them alongside the code and this README.

Recovered files, test results, and limitations of previously deleted data are listed in the [Recovery Audit Report](exam_docs/recovery_audit.md)

