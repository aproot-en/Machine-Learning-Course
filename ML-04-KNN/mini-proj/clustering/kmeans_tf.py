





## simple K-Means Clustering with TensorFlow (for beginners)
## Assign K-Means Clustering to TensorFlow (simple version for beginners)
## Updated: centroid update step to avoid empty cluster (if no member, keep centroid in place) 


import numpy as np
import tensorflow as tf


class TFKMeans:   # K-Means on TensorFlow 

    # km = TFKMeans(n_clusters=4)
    # labels = km.fit_predict(X)     # labels บอกว่าแต่ละแถวอยู่กลุ่มไหน 

    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters      # จำนวนกลุ่มที่ต้องการ
        self.max_iter = max_iter          # วนซ้ำได้สูงสุดกี่รอบ
        self.seed = seed                  # ตัวเลขสุ่ม (ตั้งไว้เพื่อให้ผลเหมือนเดิมทุกครั้ง)

    # -----------------------------------------------------------------
    def _distance(self, X, centroids):    #  X shape (n, d) , centroids shape (k, d)  ->  ผลลัพธ์ shape (n, k)

        diff = X[:, None, :] - centroids[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    # -----------------------------------------------------------------
    def fit(self, X):                     # Run K-Means until centroids are stable 
  
        X = tf.constant(X, dtype=tf.float32)
        n_samples = X.shape[0]

        # step 0 : สุ่มเลือกจุดข้อมูลมา k จุด เป็น centroid เริ่มต้น 
        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = tf.gather(X, start_idx)

        for step in range(self.max_iter):
            # step 1 : ASSIGN
            # argmin = หาว่า centroid ตัวไหนใกล้ที่สุด
            dist = self._distance(X, centroids)
            labels = tf.argmin(dist, axis=1, output_type=tf.int32)

            # step 2 : UPDATE
            # หาค่าเฉลี่ยของสมาชิกในแต่ละกลุ่ม แล้วย้าย centroid 
            new_centroids = []
            for c in range(self.n_clusters):
                members = tf.boolean_mask(X, labels == c)     # point ที่อยู่ในกลุ่ม c
                if tf.shape(members)[0] > 0:
                    new_centroids.append(tf.reduce_mean(members, axis=0))
                else:
                    new_centroids.append(centroids[c])     
            new_centroids = tf.stack(new_centroids)

            #  เช็ก stable 
            moved = float(tf.reduce_max(tf.abs(new_centroids - centroids)))
            centroids = new_centroids
            if moved < 1e-4:
                break

        # เก็บผลลัพธ์ 
        dist = self._distance(X, centroids)
        self.labels_ = tf.argmin(dist, axis=1, output_type=tf.int32).numpy()
        self.centroids_ = centroids.numpy()
        self.n_iter_ = step + 1

        # inertia = ผลรวมของ (ระยะจากแต่ละจุดถึง centroid ของตัวเอง) x 2 
        # ใช้บอกกลุ่มสั้นๆ "กระชับ" 
        self.inertia_ = float(tf.reduce_sum(tf.square(tf.reduce_min(dist, axis=1))))
        return self

# -----------------------------------------------------------------
    def fit_predict(self, X):
        return self.fit(X).labels_
