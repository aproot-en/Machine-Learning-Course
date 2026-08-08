# ML-03-Regression & Classification

Explore and experiment with regression and classification using facial image data. This module uses the same dataset for two machine learning tasks: age prediction using regression and gender classification.

The workflow starts by loading and preparing the dataset, then applies StandardScaler and PCA for preprocessing and dimensionality reduction. The processed features are used to train Ridge Regression for age prediction and Logistic Regression for gender classification. Finally, each model is evaluated using appropriate performance metrics and visualizations.

# Data

Dataset from Kaggle:

Age, Gender, and Ethnicity Face Data

Kaggle : https://www.kaggle.com/code/rashedsumon/age-gender-and-ethnicity-face-data

# Structure 
```text

ML-3-regression-and-classification/
│
├── age_gender.csv              # dataset 
├── data_loader.py              # read CSV: all 
├── main.py                     # run all 
│
├── others_dir/                
│   ├── pixels.npy
│   └── meta.csv
│
├── regression/
│   ├── main.py
│   ├── model.py                # StandardScaler → PCA → Ridge
│   ├── evaluate.py             # MAE, RMSE, R², graph
│   └── outputs/
│       ├── regression_results.png
│       └── age_samples.png
│
├── classification/
│   ├── main.py
│   ├── model.py                # StandardScaler → PCA → LogisticRegression
│   ├── evaluate.py             # accuracy, report, confusion matrix
│   └── outputs/
│       ├── confusion_matrix.png
│       └── gender_samples.png
└── requirements.txt
```

# Summary

This repository demonstrates a complete ML workflow for regression and classification using facial data. The regression task predicts age using Ridge Regression, while the classification task predicts gender using Logistic Regression. Both tasks include data preprocessing with StandardScaler and PCA, model training, evaluation, and result visualization.


