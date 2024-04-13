import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

training_image_directory = "C:/Users/Grant/Desktop/ProbeBot/ProbeBot/probebot/testingImages"  # Change this to your directory path
prompt_file_path = "C:/Users/Grant/Desktop/ProbeBot/ProbeBot/probebot/image_prompt.txt"  # Change this to your directory path
labels = {}


def loadPrompt(file_path):
    with open(prompt_file_path, "r") as file:
        prompt = file.read()
    return prompt


def userInput(plt, imageFile):
    print(imageFile)
    user_input = None
    validPrompts = ["1", "2", "3", "4", "5", "6"]

    while user_input not in validPrompts:
        user_input = input("Enter value : ")

    plt.close()
    labels[imageFile] = user_input


def save_labels_to_file(labels):
    with open("testing_data_labels.json", "w") as file:
        json.dump(labels, file)


if __name__ == "__main__":
    prompt = loadPrompt(prompt_file_path)

    training_data_labels = {}
    image_files = [
        f for f in os.listdir(training_image_directory) if f.endswith(".tif")
    ]
    image_files = sorted(image_files, key=lambda x: x.lower())
    for image_file in image_files:
        image_path = os.path.join(training_image_directory, image_file)
        image = mpimg.imread(image_path)
        imgplot = plt.imshow(image)
        plt.title(image_file)
        plt.ion()
        plt.show()
        print(prompt)
        userInput(plt, image_file)

    save_labels_to_file(labels)
