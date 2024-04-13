import numpy as np
import cv2
import os
import json
from matplotlib import pyplot as plt
from matplotlib import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import tensorflow
import traceback

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices("GPU")))
print("Num devices: ", tensorflow.config.list_physical_devices())

training_directory = "probebot/trainingImages"
training_labels_file = "training_data_labels.json"
testing_directory = "probebot/testingImages"
testing_labels_file = "testing_data_labels.json"

training_images = []
training_labels = []

testing_images = []
testing_labels = []

labels = []

print("Loading training data")
with open(training_labels_file, "r") as f:
    labels = json.load(f)

for filename in os.listdir(training_directory):
    label = labels.get(filename)
    label = int(label) - 1
    image = cv2.imread(os.path.join(training_directory, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    training_labels.append(label)
    training_images.append(cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))

print("Finished")

print("Loading testing data")

labels = []

with open(testing_labels_file, "r") as f:
    labels = json.load(f)

for filename in os.listdir(testing_directory):
    label = labels.get(filename)
    label = int(label) - 1
    image = cv2.imread(os.path.join(testing_directory, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    testing_labels.append(label)
    testing_images.append(cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))

print("Finished")

class_names = ["Pass", "Crashed", "Dirty", "Bad Etch", "Bad Angle", "Detached"]

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])

plt.show()

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(testing_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[testing_labels[i]])

plt.show()


try:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(512, 512, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)
    testing_images = np.array(testing_images)

    model.fit(
        np.array(training_images),
        np.array(training_labels),
        epochs=10,
        validation_data=(testing_images, testing_labels),
    )

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    model.save("error_detection.keras")

except BaseException as ex:
    with open("error.txt", "w") as f:
        f.write(str(ex))


# # Define CNN architecture
# model = Sequential(
#     [
#         Conv2D(
#             32,
#             kernel_size=(3, 3),
#             activation="relu",
#             input_shape=(image_height, image_width, num_channels),
#         ),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(128, activation="relu"),
#         Dense(num_classes, activation="softmax"),
#     ]
# )

# # Compile the model
# model.compile(
#     optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
# )

# # Train the model
# history = model.fit(
#     X_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=num_epochs,
#     validation_data=(X_val, y_val),
# )

# # Evaluate the model on test data
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc}")
