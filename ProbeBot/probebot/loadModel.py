from keras.models import load_model
import numpy as np
import cv2
from math import floor

import os


def percentage(val, digits):
    val *= 10 ** (digits + 2)
    return "{1:.{0}f}%".format(digits, floor(val) / 10**digits)


def predict():
    model_path = os.path.join(os.getcwd(), "error_detection.keras")

    model = load_model(model_path)

    class_names = ["Pass", "Crashed", "Dirty", "Bad Etch", "Bad Angle", "Detached"]

    image = cv2.imread("probebot/backend/temp/temp.tif")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([image]))

    index = np.argmax(prediction)

    print(f"{class_names[index]} with {percentage(prediction[0][index], 3)} confidence")
    return {class_names[index], percentage(prediction[0][index], 3)}
