from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train, pca_components=150, alpha=1.0):
    # One pipeline, so the test set always gets the same transform
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=42),
        Ridge(alpha=alpha),
    )

    model.fit(X_train, y_train)

    return model


def predict_model(model, X_test):
    # Ridge can predict negative ages, so clip to the real range
    return model.predict(X_test).clip(1, 116)
