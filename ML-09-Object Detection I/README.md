# ML-09 Object Detection I

This project explores **R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN** using Python and TensorFlow/Keras with the BDD100K dataset. It provides simplified educational baselines for model training, evaluation, image testing, and real-time webcam testing.

## Dataset

**BDD100K — YOLO Format**

Kaggle Link: https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download

Download the dataset and organize it into `train`, `val`, and `test` directories, each containing `images` and `labels` folders, along with the `data.yaml` configuration file.

# Structure

```text
ML-09-Object-Detection-I/
├── bdd100k/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
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
├── 03-Faster-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── 04-Mask-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── requirements.txt
└── README.md
```

## Project Structure

Each architecture follows the same workflow:

- `train.py` — Train the model.
- `evaluate.py` — Evaluate model performance.
- `image_test.py` — Test the trained model on images.
- `realtime_test.py` — Run real-time testing using a webcam.
- `outputs/` — Store trained models and results.

Each folder is standalone and uses the shared BDD100K dataset. Model creation, data loading, and preprocessing are implemented directly in the Python scripts. No additional `_core.py`, `common/`, or `network/` modules are required.

## Important Notes

All four implementations currently use educational **CNN crop-classification baselines**, rather than complete R-CNN, Fast R-CNN, Faster R-CNN, or Mask R-CNN architectures.

Image and webcam testing use sliding-window classification. Evaluation supports Accuracy, Precision, Recall, and F1-score, but not detection mAP.

A complete Mask R-CNN implementation also requires instance segmentation masks, which are not included in standard YOLO bounding-box annotations.

These implementations are intended for educational purposes and should not be used for direct comparisons of full object detection architectures.

# Summary

This project explores **R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN** using Python and TensorFlow/Keras with the **BDD100K dataset**. It provides educational CNN-based crop-classification baselines with a simplified workflow for data loading, preprocessing, model training, evaluation, image testing, and real-time webcam testing. Each architecture is organized into a separate folder with four main Python scripts. The current implementations demonstrate object classification concepts rather than complete object detection or instance segmentation architectures.


