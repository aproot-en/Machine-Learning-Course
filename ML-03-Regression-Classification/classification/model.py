from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train, pca_components=150, C=1.0):
    # One pipeline, so the test set always gets the same transform.
    # PCA also keeps LogisticRegression from stalling on 2,304 features.
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=42),
        LogisticRegression(C=C, max_iter=1000),
    )

    model.fit(X_train, y_train)

    return model


def predict_model(model, X_test):

    return model.predict(X_test)
