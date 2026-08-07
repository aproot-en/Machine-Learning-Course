# ML-08-Deep Convolutional Neural Network (DCNN)

Build a DCNN based on the VGG architecture using Python for image recognition. The project covers image loading, preprocessing, dataset splitting, DCNN (VGG) model training, evaluation, and prediction for Cat and Dog image classification

# Dataset
Kaggle Dogs vs. Cats: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

# Project Structure

```text
ML-08-DNN/
│
├── PetImages/                     
│   ├── Cat/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   └── Dog/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
│
├── classification/
│   ├── main.py                     # Main training pipeline
│   ├── data_loader.py              # Load images and skip corrupted files
│   ├── preprocessing.py            # Resize images and convert BGR to RGB
│   ├── split_data.py               # Split the dataset into training, validation, and test sets
│   ├── vgg_model.py                # Build, train, save, and predict using the VGG model
│   ├── evaluate.py                 # Accuracy, classification report, confusion matrix, and training history plots
│   ├── test_vgg.py                 # Test the trained model using four random images 
│   │
│   └── outputs/                    # Generated files 
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── vgg_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
│
└── requirements.txt

```

# Summary

This project implements a VGG-based DCNN for Cat and Dog image classification. It includes image loading, preprocessing, dataset splitting, model training, evaluation, and prediction using a modular Python pipeline.

