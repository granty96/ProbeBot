import torch
import torch.nn.functional as F
from probebot.trainModelTorch import CNNModel
from torchvision import transforms
from math import floor
from PIL import Image
import os


def percentage(val, digits):
    val *= 10 ** (digits + 2)
    return "{1:.{0}f}%".format(digits, floor(val) / 10**digits)


def predict():
    model_path = os.path.join(os.getcwd(), "error_detection.pth")

    # Load the PyTorch model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    class_names = ["Pass", "Crashed", "Dirty", "Bad Etch", "Bad Angle", "Detached"]

    # Load and preprocess the image
    image_path = "probebot/backend/temp/temp.tif"
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((512, 512))

    # Transform the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        prediction = model(image)

    # Convert prediction to probabilities using softmax
    probabilities = F.softmax(prediction, dim=1)
    confidence, index = torch.max(probabilities, 1)

    print(
        f"{class_names[index.item()]} with {percentage(confidence.item(), 3)} confidence"
    )
    return [class_names[index.item()], percentage(confidence.item(), 3)]
