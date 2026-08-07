# ML-06-Neural Network (NN)

Build a simple NN pipeline using Python for image recognition. The project covers image loading, preprocessing, dataset splitting, neural network training, evaluation, and prediction.

# Data 
Cats and Dogs Classification Dataset: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

# Structure

```text
ML-06-NN/
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
│   ├── nn_model.py                 # Build, train, save, and predict with the Neural Network
│   ├── evaluate.py                 # Accuracy, classification report, confusion matrix, and training plots
│   ├── test_nn.py                  # Test the trained model using four random images
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
│       ├── nn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
│
└── requirements.txt
```
# Summary
The project uses a NN for Cat and Dog image recognition. Images are automatically loaded from the dataset directories, resized to a fixed resolution, and converted from BGR to RGB format during preprocessing. The dataset is then split into training, validation, and test sets before being used to train the neural network. The trained model is evaluated using accuracy, precision, recall, F1-score, a confusion matrix, and training history plots to assess its classification performance.


