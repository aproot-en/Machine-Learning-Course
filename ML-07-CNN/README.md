# ML-07-CNN

Build a Convolutional Neural Network (CNN) pipeline using Python for image recognition. The project covers image loading, preprocessing, dataset splitting, CNN model training, evaluation, and prediction.

# Dataset

Kaggle Dogs vs. Cats: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
# Project Structure

```text
ML-07-CNN/
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
│   ├── split_data.py               # Split data into training, validation, and test sets
│   ├── cnn_model.py                # Build, train, save, and predict with the CNN model
│   ├── evaluate.py                 # Accuracy, classification report, confusion matrix, and training plots
│   ├── test_cnn.py                 # Test the trained model using four random images
│   └── outputs/                    
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── cnn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
└── requirements.txt
```

The project uses a Convolutional Neural Network (CNN) for Cat and Dog image recognition. Images are automatically loaded from the dataset directories, resized to a fixed resolution, and converted from BGR to RGB format during preprocessing. The dataset is then split into training, validation, and test sets before being used to train the CNN model. The trained model is evaluated using accuracy, precision, recall, F1-score, a confusion matrix, and training history plots to assess its classification performance.
