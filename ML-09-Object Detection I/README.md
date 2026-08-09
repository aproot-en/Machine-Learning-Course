# ML-09-Object Detection I
Learn and implement region-based object detection using R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN with Python. The project covers region proposals, CNN feature extraction, RoI Pooling, Region Proposal Networks (RPN), anchor boxes, bounding box regression, Non-Maximum Suppression (NMS), RoI Align, object detection, and instance segmentation.

-----------
# Data
The dataset is designed for face mask detection and contains **853 images** across **3 classes**, with object annotations provided as bounding boxes in **PASCAL VOC format**. The three classes are **With Mask**, **Without Mask**, and **Mask Worn Incorrectly**.

Kaggle Link: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection 

# Structure

```text
ML-09-Object-Detection-I/
│
├── face-mask-data/                 
│   ├── images/                     
│   └── annotations/               
│
├── 01-R-CNN/
│   ├── main_test.py                # Test with a custom image
│   ├── load_data.py                # Load images and XML annotations
│   ├── model.py                    # CNN for crop classification
│   ├── training_model.py           # Model training
│   ├── utils.py                    # IoU, NMS, Selective Search, bounding box drawing
│   └── outputs/                    # Trained models and detection results
│
├── 02-Fast-R-CNN/
│   ├── main_test.py                # Test with a custom image
│   ├── load_data.py                # Load images
│   ├── model.py                    # Backbone + RoI Pooling + two heads
│   ├── training_model.py           # Model training
│   ├── utils.py                    # Utility functions
│   └── outputs/                    # Trained models and detection results
│
├── 03-Faster-R-CNN/
│   ├── main_test.py                # Test with a custom image
│   ├── load_data.py                # Load images and XML annotations
│   ├── model.py                    # Backbone + RPN + detector
│   ├── anchors.py                  # Generate anchors and encode/decode deltas
│   ├── roi_pooling.py              # RoI Pooling
│   ├── training_model.py           # Model training
│   ├── utils.py                    # Utility functions
│   └── outputs/                    # Trained models and detection results
│
├── 04-Mask-R-CNN/
│   ├── main_test.py                # Test with a custom image
│   ├── load_data.py                # Load images and XML annotations
│   ├── model.py                    # Backbone + RPN + detector + mask head
│   ├── anchors.py                  # Generate anchors and encode-decode deltas
│   ├── roi_pooling.py              # RoI Pooling
│   ├── training_model.py           # Model training
│   ├── utils.py                    # Utility functions
│   └── outputs/                    # Trained models and detection results
│
└── requirements.txt
```

# Summary
This project implements **R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN** for face mask object detection. It includes image loading, preprocessing, region proposal, model training, evaluation, object detection, and instance segmentation using a modular Python pipeline.


