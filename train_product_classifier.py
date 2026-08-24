"""Part 2: transfer learning on the canonical Fashion-MNIST dataset."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

from product_classifier import CLASS_NAMES, ProductClassifier, image_transform


ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data"
CACHE_PATH = DATA_ROOT / "fashion_mnist_resnet18_features.pt"
MODEL_PATH = ROOT / "models" / "product_classifier.pt"
REPORT_PATH = ROOT / "product_classifier_report.json"
SEED = 42
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 10_000
IMAGE_BATCH_SIZE = 128
HEAD_BATCH_SIZE = 512
HEAD_LEARNING_RATE = 1e-3
HEAD_EPOCHS = 20
FINETUNE_LEARNING_RATE = 1e-5
FINETUNE_EPOCHS = 3


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def extract_features(backbone, loader, device):
    backbone.eval()
    feature_batches, label_batches = [], []
    for batch_number, (images, labels) in enumerate(loader, start=1):
        feature_batches.append(backbone(images.to(device)).cpu())
        label_batches.append(labels)
        if batch_number % 50 == 0:
            print(f"  extracted {batch_number * loader.batch_size:,} images")
    return torch.cat(feature_batches), torch.cat(label_batches)


def accuracy_from_logits(logits, labels):
    return float((logits.argmax(1) == labels).float().mean())


def evaluate_head(head, loader, device):
    head.eval()
    correct = total = 0
    with torch.inference_mode():
        for features, labels in loader:
            labels = labels.to(device)
            predictions = head(features.to(device)).argmax(1)
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / total


def train_head(model, train_features, train_labels, val_features, val_labels, device):
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels), batch_size=HEAD_BATCH_SIZE
    )
    head = model.classifier.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_accuracy, best_state = 0.0, None
    history = []
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        val_accuracy = evaluate_head(head, val_loader, device)
        history.append(val_accuracy)
        print(f"head epoch {epoch:02d}: validation accuracy={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return best_accuracy, history


def image_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / total


def fine_tune_layer4(model, train_loader, val_loader, device):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=FINETUNE_LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = image_accuracy(model, val_loader, device)
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        # Frozen BatchNorm running statistics must remain fixed.
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        val_accuracy = image_accuracy(model, val_loader, device)
        history.append(val_accuracy)
        print(f"fine-tune epoch {epoch:02d}: validation accuracy={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_accuracy, history


@torch.inference_mode()
def predict_all(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for images, labels in loader:
        predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
        actual.extend(labels.tolist())
    return np.asarray(actual), np.asarray(predicted)


def export_test_samples(raw_test):
    output_dir = DATA_ROOT / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    for index, (_, label) in enumerate(raw_test):
        if label not in selected:
            selected[label] = index
        if len(selected) == len(CLASS_NAMES):
            break
    paths = []
    for label, index in sorted(selected.items()):
        image, _ = raw_test[index]
        safe_name = CLASS_NAMES[label].lower().replace("/", "_").replace(" ", "_")
        path = output_dir / f"{index:05d}_{safe_name}.png"
        image.save(path)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def run():
    seed_everything()
    device = choose_device()
    print(f"device: {device}")
    transform = image_transform()
    full_train = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_data = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
    raw_test = FashionMNIST(DATA_ROOT, train=False, download=False)
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train)),
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=np.asarray(full_train.targets),
    )
    assert len(train_indices) == TRAIN_SIZE
    assert len(val_indices) == VALIDATION_SIZE
    assert len(test_data) == TEST_SIZE
    train_subset, val_subset = Subset(full_train, train_indices), Subset(full_train, val_indices)
    loader_args = dict(batch_size=IMAGE_BATCH_SIZE, num_workers=0)
    train_image_loader = DataLoader(train_subset, shuffle=False, **loader_args)
    val_image_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    model = ProductClassifier(pretrained=True).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, map_location="cpu")
        train_features, train_labels = cache["train_features"], cache["train_labels"]
        val_features, val_labels = cache["val_features"], cache["val_labels"]
        print(f"loaded cached features from {CACHE_PATH}")
    else:
        print("extracting frozen ResNet-18 features once")
        train_features, train_labels = extract_features(
            model.backbone, train_image_loader, device
        )
        val_features, val_labels = extract_features(model.backbone, val_image_loader, device)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "val_features": val_features,
                "val_labels": val_labels,
            },
            CACHE_PATH,
        )

    feature_val_accuracy, head_history = train_head(
        model,
        train_features,
        train_labels,
        val_features,
        val_labels,
        device,
    )
    fine_tuning_required = feature_val_accuracy < 0.80
    fine_tune_history = []
    final_val_accuracy = feature_val_accuracy
    if fine_tuning_required:
        # Recreate shuffled training loader only if gradual unfreezing is needed.
        train_image_loader = DataLoader(
            train_subset,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
            **loader_args,
        )
        final_val_accuracy, fine_tune_history = fine_tune_layer4(
            model, train_image_loader, val_image_loader, device
        )

    # The test set is evaluated only after validation-based model selection.
    y_true, y_pred = predict_all(model, test_loader, device)
    test_accuracy = float((y_true == y_pred).mean())
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    off_diagonal = [
        (int(matrix[a, p]), CLASS_NAMES[a], CLASS_NAMES[p])
        for a in range(len(CLASS_NAMES))
        for p in range(len(CLASS_NAMES))
        if a != p
    ]
    largest_confusions = sorted(off_diagonal, reverse=True)[:10]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    sample_paths = export_test_samples(raw_test)
    report = {
        "source": "torchvision.datasets.FashionMNIST (Zalando Research)",
        "splits": {"train": TRAIN_SIZE, "validation": VALIDATION_SIZE, "test": TEST_SIZE},
        "preprocessing": {
            "channels": 3,
            "image_size": [224, 224],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "training": {
            "backbone": "ImageNet-pretrained ResNet-18",
            "image_batch_size": IMAGE_BATCH_SIZE,
            "head_batch_size": HEAD_BATCH_SIZE,
            "optimizer": "Adam",
            "head_learning_rate": HEAD_LEARNING_RATE,
            "head_epochs": HEAD_EPOCHS,
            "head_validation_history": head_history,
            "feature_extraction_validation_accuracy": feature_val_accuracy,
            "fine_tuning_required": fine_tuning_required,
            "fine_tune_learning_rate": FINETUNE_LEARNING_RATE,
            "fine_tune_epochs": len(fine_tune_history),
            "fine_tune_validation_history": fine_tune_history,
            "final_validation_accuracy": final_val_accuracy,
        },
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": per_class[name]["precision"],
                "recall": per_class[name]["recall"],
                "f1-score": per_class[name]["f1-score"],
                "support": int(per_class[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "largest_directional_confusions": largest_confusions,
        "sample_images": sample_paths,
        "artifact": str(MODEL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
