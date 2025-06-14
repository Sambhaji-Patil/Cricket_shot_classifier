import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "model/epoch_15.pth"

class_names = ['Drive','Legglance Flick','Pull Shot','Sweep_Shot']  # Update with your actual class names

# Define model once
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, len(class_names))
)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

    max_prob = np.max(probabilities)
    predicted_indices = [i for i, p in enumerate(probabilities) if (max_prob - p) <= 0.05]
    predicted_class = [class_names[i] for i in predicted_indices]
    result = {class_names[i]: round(probabilities[i] * 100, 2) for i in range(len(class_names))}

    return predicted_class, result
