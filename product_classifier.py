"""Loading and single-image inference for the Fashion-MNIST classifier."""

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_PATH = Path(__file__).with_name("models") / "product_classifier.pt"
PRETRAINED_WEIGHTS_PATH = (
    Path(__file__).with_name("data") / "pretrained" / "resnet18-f37072fd.pth"
)


def image_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ProductClassifier(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained and not PRETRAINED_WEIGHTS_PATH.exists():
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)
            if pretrained:
                backbone.load_state_dict(
                    torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
                )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(self, images):
        return self.classifier(self.backbone(images))


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = "cpu",
) -> ProductClassifier:
    """Load the saved state dict without downloading pretrained weights."""
    model = ProductClassifier(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


@torch.inference_mode()
def classify_product_image(
    image_path: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = "cpu",
) -> Tuple[str, float]:
    """Return (predicted class name, confidence) for one image file."""
    model = load_model(model_path, device)
    with Image.open(image_path) as image:
        tensor = image_transform()(image.convert("L")).unsqueeze(0).to(device)
    probabilities = model(tensor).softmax(dim=1)[0]
    class_index = int(probabilities.argmax())
    return CLASS_NAMES[class_index], float(probabilities[class_index])
