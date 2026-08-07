import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2):
    # y must be an array, not a list, for stratify to work
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
