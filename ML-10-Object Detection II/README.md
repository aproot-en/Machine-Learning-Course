# ML-10-Object Detection II

This project implements **YOLOv8-based object detection** using the Berkeley DeepDrive (BDD100K) dataset. It detects road objects such as vehicles, pedestrians, and traffic infrastructure, and covers training, evaluation, image inference, and real-time webcam detection.

## 1. Project Overview

The project fine-tunes **Ultralytics YOLOv8n** with pretrained weights on the BDD100K dataset.

**Features:**
- Detection across 10 road-object classes
- Pretrained YOLOv8 fine-tuning
- Evaluation using Precision, Recall, mAP50, and mAP50-95
- Image and real-time webcam inference
- Automatic model checkpoints and epoch-by-epoch CSV training logs

## 2. Dataset

The [BDD100K dataset](https://bdd-data.berkeley.edu/) is used in YOLO annotation format.

| Split | Images |
|---|---:|
| Training | 70,000 |
| Validation | 10,000 |
| Test | Provided in the dataset (count not specified) |

**Classes:** `person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`.

Each YOLO label line contains a class ID and normalized bounding-box coordinates.

## 3. Project Structure

```text
YOLO-BDD100K/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── train.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md
```

| Script | Purpose |
|---|---|
| `train.py` | Train YOLOv8 and save checkpoints and training logs |
| `evaluate.py` | Evaluate on labeled test images |
| `image_test.py` | Detect objects in randomly selected test images |
| `realtime_test.py` | Run webcam-based real-time detection |


## 4. Object Detection

**Image detection** — randomly select test images, draw boxes and labels, and save annotated results:

```bash
python image_test.py
```

**Real-time detection** — open webcam index 0 and display annotated frames. Press **Q** to exit:

```bash
python realtime_test.py
```

The inference scripts must point to the actual trained `best.pt` path. For evaluation, the test split requires corresponding label files.

## 5. Future Work

Potential improvements include larger YOLO models, hyperparameter tuning, improved small-object detection, class-wise error analysis, inference FPS measurement, and web deployment.

## 6. References

- [BDD100K / Berkeley DeepDrive](https://bdd-data.berkeley.edu/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/)


