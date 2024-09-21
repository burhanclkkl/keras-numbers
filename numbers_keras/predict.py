import pandas as # The prediction step provide correct image path
import numpy as np
import cv2
from nnet import Neuralnetwork

def read_image(path):
    img = cv2.imread(path.cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
    img = img.reshape(1,28,28,1)
    return img

okunanResim = read_image("")
ai = Neuralnetwork()
