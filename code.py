import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import pickle

data_path = "<enter path to dataset>"
labels_path = "<enter path for labels>"

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

features = []
labels = []

with open(labels_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        image_path, label = line.strip().split(',')
        image = load_img(data_path + image_path, target_size=(224, 224))
        print(image_path)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = base_model.predict(image)
        features.append(image.flatten())
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm = SVC(kernel= 'poly')

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(y_pred)
print(y_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

filename = 'newspaper_classifier_model_'+ str(accuracy) + '.sav'
pickle.dump(svm, open(filename, 'wb'))

