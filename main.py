import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Datasets size")
print("Train data:", X_train.shape)
print("Test data:", X_test.shape)

images_train =  []
for image_train in X_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in X_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

neural_network = MLPClassifier(hidden_layer_sizes=(200,100,50),random_state=1)

neural_network.fit(StandardScaler().fit_transform(images_train), y_train)

print("KONIEC")