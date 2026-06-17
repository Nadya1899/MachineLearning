import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Dataset
X_train = np.array([[3, 2],
                    [9, 5],
                    [6, 6],
                    [4, 7],
                    [9, 9],
                    [14, 13],
                    [12, 15],
                    [16, 14]])

y_train = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'])

# Membuat model KNN
k = 7  # jumlah tetangga terdekat yang akan digunakan
knn = KNeighborsClassifier(n_neighbors=k)

# Melatih model
knn.fit(X_train, y_train)

# Data uji
X_test = np.array([[14, 8],
                   [5, 5]])
y_test = np.array(['A', 'A'])

# Memprediksi label dari data uji
y_pred = knn.predict(X_test)

# Menampilkan hasil prediksi
for i, pred in enumerate(y_pred):
    print("Data uji", i+1, ":", X_test[i], "Prediksi label:", pred)


# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)