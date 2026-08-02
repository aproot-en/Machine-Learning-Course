









import numpy as np
import tensorflow as tf


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k          # จำนวนเพื่อนบ้านที่ใช้

    # -----------------------------------------------------------------
    def fit(self, X, y):    #Train KNN 
       
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.int32)
        self.n_classes = int(y.max()) + 1
        return self

    # -----------------------------------------------------------------
    def _distance(self, X_new):
        """
        distance = sqrt( (x1-y1)² + (x2-y2)² + ... )

        """
        diff = X_new[:, None, :] - self.X_train[None, :, :]     # ผลต่างของทุกคู่
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))  # (n_new, n_train)

    # -----------------------------------------------------------------
    def predict(self, X):
        """predict class of new data and return array of class labels"""
        X = tf.constant(X, dtype=tf.float32)

        # step 1 : distance
        dist = self._distance(X)

        # step 2 : select k 
        # tf.math.top_k หา "find max value" 
        _, idx = tf.math.top_k(-dist, k=self.k)          # idx = ตำแหน่งของเพื่อนบ้าน
        neighbor_labels = tf.gather(self.y_train, idx)   # คลาสของเพื่อนบ้าน (n_new, k)

        # step 3 : vote k 
        onehot = tf.one_hot(neighbor_labels, depth=self.n_classes)
        votes = tf.reduce_sum(onehot, axis=1)            # (n_new, n_classes)

        return tf.argmax(votes, axis=1).numpy()          # คลาสที่ได้คะแนนมากสุด

    # -----------------------------------------------------------------
    def score(self, X, y):
        """calculate accuracy = proportion of correct predictions"""
        return float(np.mean(self.predict(X) == y))
