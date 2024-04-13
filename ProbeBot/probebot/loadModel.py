from keras.models import load_model
import numpy as np
import cv2
from math import floor


def percentage(val, digits):
    val *= 10 ** (digits + 2)
    return "{1:.{0}f}%".format(digits, floor(val) / 10**digits)


model = load_model("error_detection.keras")
class_names = ["Pass", "Crashed", "Dirty", "Bad Etch", "Bad Angle", "Detached"]

image = cv2.imread("probebot/testingImages/4991_001.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

prediction = model.predict(np.array([image]))

index = np.argmax(prediction)
# print(f'Prediction is "{class_names[index]}" with {prediction[index]} confidence.')
print(f"{class_names[index]} with {percentage(prediction[0][index], 3)} confidence")
