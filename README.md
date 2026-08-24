# Part 2 — Product Image Categoriser

This project classifies Flipkart-style apparel, footwear, and accessories using
transfer learning on the canonical
[Zalando Research Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
It uses `torchvision.datasets.FashionMNIST` without substituting or simulating
data.

## Reproduce training

PyTorch 2.2 requires Python 3.9–3.12. The run reported below used Python 3.9.6:

```bash
python3 -m venv .venv39
.venv39/bin/pip install -r requirements.txt
.venv39/bin/python train_product_classifier.py
```

The script downloads Fashion-MNIST and the official ImageNet-pretrained
ResNet-18 weights automatically. It caches frozen backbone features in `data/`
so future head-training runs do not repeat the expensive CNN forward pass.

## Dataset and untouched split

The standard 60,000-image Fashion-MNIST training split was divided with a
stratified split (`random_state=42`); the official test set was not passed
through the model until validation-based model selection was finished.

| Split | Images | Purpose |
|---|---:|---|
| Training | 55,000 | Fit the classifier head |
| Validation | 5,000 | Select the best head epoch and decide on fine-tuning |
| Test | 10,000 | One final evaluation only |

## Preprocessing and transfer learning

Each 28×28 grayscale image is replicated to three channels, resized to
**224×224**, converted to a tensor, and normalized with the ImageNet statistics
used for ResNet-18: mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

The early and middle ResNet-18 backbone layers were frozen. Its 512-dimensional
output was extracted and cached once for each training and validation image.
A new `512 → 256 → ReLU → Dropout(0.2) → 10` classifier head was trained with:

- Optimizer: Adam
- Learning rate: 0.001
- Image extraction batch size: 128
- Cached-feature head batch size: 512
- Epochs: 20

**Feature extraction alone was sufficient.** Its best validation accuracy was
**91.58%** at epoch 19, above the 80% fine-tuning trigger. Consequently no
backbone layer was unfrozen: validation accuracy before fine-tuning was 91.58%
and after the fine-tuning decision remained **91.58% (no fine-tuning run)**.
Had it fallen below 80%, the script would have unfrozen only ResNet `layer4`
for three Adam epochs at the lower learning rate `1e-5`.

## Held-out test evaluation

Final accuracy on the untouched 10,000-image test set was **90.43%**.

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T-shirt/top | 87.92% | 83.00% | 85.39% | 1,000 |
| Trouser | 99.18% | 97.30% | 98.23% | 1,000 |
| Pullover | 89.80% | 85.40% | 87.54% | 1,000 |
| Dress | 86.38% | 92.60% | 89.38% | 1,000 |
| Coat | 85.48% | 84.20% | 84.84% | 1,000 |
| Sandal | 97.35% | 95.50% | 96.42% | 1,000 |
| Shirt | 71.44% | 75.80% | 73.56% | 1,000 |
| Sneaker | 94.19% | 95.70% | 94.94% | 1,000 |
| Bag | 98.70% | 98.50% | 98.60% | 1,000 |
| Ankle boot | 95.25% | 96.30% | 95.77% | 1,000 |

### Confusion matrix

Rows are actual classes and columns are predicted classes, in this order:
`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`,
`Sneaker`, `Bag`, `Ankle boot`.

| Actual \ Predicted | T-shirt | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Boot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-shirt/top | 830 | 2 | 12 | 28 | 3 | 1 | 119 | 0 | 4 | 1 |
| Trouser | 3 | 973 | 1 | 17 | 1 | 1 | 4 | 0 | 0 | 0 |
| Pullover | 10 | 0 | 854 | 12 | 48 | 0 | 75 | 0 | 1 | 0 |
| Dress | 9 | 5 | 7 | 926 | 21 | 0 | 31 | 0 | 1 | 0 |
| Coat | 0 | 1 | 44 | 42 | 842 | 1 | 67 | 0 | 3 | 0 |
| Sandal | 0 | 0 | 0 | 0 | 0 | 955 | 0 | 32 | 0 | 13 |
| Shirt | 91 | 0 | 33 | 43 | 70 | 0 | 758 | 0 | 4 | 1 |
| Sneaker | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 957 | 0 | 33 |
| Bag | 1 | 0 | 0 | 4 | 0 | 3 | 7 | 0 | 985 | 0 |
| Ankle boot | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 27 | 0 | 963 |

The largest confusion is **T-shirt/top → Shirt (119 images)**, with another 91
shirts classified as T-shirts. At only 28×28 grayscale resolution, both have a
short-sleeved torso silhouette and a similar neckline. The primary distinction
is subtle structure such as a collar, placket, or sleeve cut, which can occupy
only a few pixels and is further softened by resizing.

The next distinct pair is **Pullover → Shirt (75 images)**; 33 shirts also went
the other direction. Both appear as upper-body garments with sleeves and a
central torso block. Fashion-MNIST removes color and fabric texture, so sleeve
length, neckline, and garment thickness are the remaining cues; pose and loose
fits can make those silhouettes overlap substantially.

A third strong pair is **Shirt ↔ Coat**: 70 shirts were predicted as coats and
67 coats as shirts. Both commonly have long sleeves and an open or structured
front. In a small monochrome silhouette, coat length, lapels, and heavier fabric
are weak signals, especially when the photographed item is cropped similarly.

All matrix values and confusion pairs above come from the saved model's real
predictions and are also stored in `product_classifier_report.json`.

## Saved artifact and one-image prediction

The trained weights are saved at `models/product_classifier.pt`. The reusable
loader and inference function are implemented in `product_classifier.py`; they
construct the architecture without downloading backbone weights, load the
state dict, apply the same preprocessing, and call the model's real softmax
output.

```python
from product_classifier import classify_product_image

label, confidence = classify_product_image(
    "data/sample_images/00009_sneaker.png"
)
print(label, confidence)
```

Part 3's `classify_product_image(image_path)` tool should call this exact
function rather than use a hardcoded category.

## Exported real test images

`data/sample_images/` contains ten actual PNG files exported directly from the
official Fashion-MNIST test split—one per class. Their true labels are embedded
in their filenames:

- `00019_t-shirt_top.png`
- `00002_trouser.png`
- `00001_pullover.png`
- `00013_dress.png`
- `00006_coat.png`
- `00008_sandal.png`
- `00004_shirt.png`
- `00009_sneaker.png`
- `00018_bag.png`
- `00000_ankle_boot.png`
