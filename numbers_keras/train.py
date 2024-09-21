import pandas as pd
import numpy as np
from nnet import Neuralnetwork

def category_encode(labels, label_tye= 10):
    category_label = []
    for i in labels:
        vector = np.zeros(label_tye)
        vector[i] = 1
        category_label.append(vector)
    return np.array(category_label)

def optimizasyon(X):
    X = X / 255
    X_matrix = []
    size = len(X)
    for i in range(size):
        X_matrix.append(np.array(X.iloc[i]).reshape(( 28, 28, 1)))
    return np.array(X_matrix)

dataset = pd.read_csv("dataset.csv")
y = dataset["label"]
y = category_encode(y)
X = dataset.drop("label", axis=1)
X = optimizasyon(X)

def egitimVerisi(): pass

egitimVerisi.X = X
egitimVerisi.y = y
yapayzeka = Neuralnetwork()
yapayzeka.egit(egitimVerisi,1)
yapayzeka.kaydet("model_1.keras")
