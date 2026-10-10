# ML-09-Object Detection I

This project explores **R-CNN, and Fast R-CNN** using Python and TensorFlow/Keras with the Berkeley DeepDrive (BDD100K) dataset. It covers model training, evaluation, image inference, and real-time webcam testing through simplified educational implementations.

## 1. Project Overview

The project introduces four CNN-based object detection architectures using a shared BDD100K dataset. Each architecture is organized into a separate folder with independent scripts for training, evaluation, and inference.

**Features:**
- Exploration of four R-CNN-based architectures
- Shared BDD100K dataset with 10 road-object classes
- Model training and evaluation
- Image and real-time webcam testing
- Model checkpoints and experiment outputs
- Separate implementations for architectural comparison

## 2. Dataset

The [BDD100K dataset](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download) is used in YOLO annotation format.

| Split | Images |
|---|---:|
| Training | 70,000 |
| Validation | 10,000 |
| Test | Provided in the dataset (count not specified) |

**Classes:** `person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`.

The dataset is shared across all four architectures. YOLO annotations can be converted into the required bounding-box format during preprocessing.

## 3. Project Structure

```text
ML-09-Object-Detection-I/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
│
├── 01-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── 02-Fast-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── requirements.txt
└── README.md
```

| Script | Purpose |
|---|---|
| `train.py` | Train the model and save checkpoints |
| `evaluate.py` | Evaluate model performance on labeled images |
| `image_test.py` | Test the trained model on selected images |
| `realtime_test.py` | Run webcam-based real-time inference |
| `outputs/` | Store trained models, logs, and inference results |

Each architecture uses the same dataset and follows an independent workflow. Model creation, data loading, and preprocessing are implemented directly in the Python scripts.

## 4. Object Detection

**Model training** — navigate to the selected architecture folder and run:

```bash
python train.py --limit 200 --epochs 50
```

**Model evaluation** — evaluate the trained model using labeled test images:

```bash
python evaluate.py --limit 200 --split test
```

**Image testing** — select test images, run inference, and save prediction results:

```bash
python image_test.py
```

**Real-time testing** — open webcam index 0 and perform real-time inference. Press **Q** to exit:

```bash
python realtime_test.py --camera 0 --threshold 0.85 --regions 25
```

The scripts must reference the correct dataset directory and trained model checkpoint.

**Implementation note:** The current R-CNN and Fast R-CNN implementations use simplified crop-classification baselines rather than complete object detection architectures. Full implementations require region proposals, bounding-box regression, and detection postprocessing. The BDD100K dataset provides YOLO-format bounding-box annotations for training and evaluation.

## 5. References

- [BDD100K / Berkeley DeepDrive](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download)
- [R-CNN](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
