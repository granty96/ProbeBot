import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import json


# Define the model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model():
    model_path = "error_detection.pth"
    if os.path.exists(model_path):
        model = CNNModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        return model
    else:
        # Train the model
        training_directory = "probebot/trainingImages"
        training_labels_file = "training_data_labels.json"
        training_images, training_labels = load_images_and_labels(
            training_directory, training_labels_file
        )
        train_dataset = ImageDataset(
            training_images, training_labels, transform=transform
        )
        train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = CNNModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}"
            )

        # Save the model
        torch.save(model.state_dict(), model_path)

        model.eval()  # Set model to evaluation mode
        return model


def load_images_and_labels(directory, labels_file):
    images = []
    labels = []
    with open(labels_file, "r") as f:
        label_mapping = json.load(f)
    for filename in os.listdir(directory):
        label = label_mapping.get(filename)
        label = int(label) - 1
        image = Image.open(os.path.join(directory, filename))
        image = image.convert("RGB")
        image = image.resize((512, 512))
        images.append(image)
        labels.append(label)
    return images, labels


class ImageDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load and preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def predict(model, image_path):
    class_names = ["Pass", "Crashed", "Dirty", "Bad Etch", "Bad Angle", "Detached"]

    # Preprocess the image
    image = preprocess_image(image_path)

    # Make prediction
    with torch.no_grad():
        prediction = model(image)

    # Convert prediction to probabilities using softmax
    probabilities = torch.softmax(prediction, dim=1)
    confidence, index = torch.max(probabilities, 1)

    return class_names[index.item()], confidence.item()


if __name__ == "__main__":
    model = load_model()
    image_path = "probebot/backend/temp/temp.tif"
    result_class, result_confidence = predict(model, image_path)
    print(f"Predicted class: {result_class}, Confidence: {result_confidence}")
