# Space Sponge

> A computer vision pipeline that pulls planets out of images, cleans them up, and teaches a neural network to recognise them.

## What is this?

A project that tackles planet classification end-to-end. The pipeline does three things:

1. **Extracts planets** from their black backgrounds using classical computer vision (Hough Circle Transform)
2. **Generates training data** through aggressive augmentation (Gaussian blur, brightness shifts, pixelation). It turns ~1,300 images into ~61,000
3. **Trains a classifier** using transfer learning on MobileNetV2 to recognise 9 classes

The result is a model that achieves **99.95% test accuracy** classifying Earth, Moon, Jupiter, Mars, Mercury, Neptune, Pluto, Uranus, and Venus.

This was built to feed into a robotics system. The idea being a robot could point its camera at a window showing a planet and identify what it is.

## The Interesting Bits

### Hough Circle Transform for Background Removal

The source images are planets rendered on pure black backgrounds. I used Hough Transform using:

```python
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)
```

It creates a circular mask at the detected circle's position, apply it.

### Data Augmentation

Starting with ~1,300 original images (149 each for Earth/Moon, 21 each for other planets), the augmentation pipeline generates:

- 9 Gaussian blur variants per image (kernel sizes 5×5 to 37×37)
- 4 brightness variants (0.5x, 0.75x, 1.25x, 1.5x)
- 5 pixelation levels
- All combinations of pixelation + blur

This balloons to ~61,000 training samples, giving the model plenty of variation to generalise from.

### Transfer Learning with MobileNetV2

Using a pre-trained MobileNetV2 and fine-tuning only the classifier head:

```python
base_model = models.mobilenet_v2()
for param in base_model.parameters():
    param.requires_grad = True  # Fine-tune everything

num_features = base_model.classifier[1].in_features
base_model.classifier = nn.Linear(num_features, len(classes))
```

The model converges fast as it only takes 2 epochs to hit 99.97% validation accuracy. The pre-trained features from ImageNet transfers well but might be overfitting to the synthetic nature of the dataset.

## Project Structure

```
.
├── extracting.ipynb      # Background removal (Hough Circles)
├── extracting.py         # Same as notebook, standalone script
├── manipulation.py        # Data augmentation pipeline
├── model.ipynb           # Training and evaluation
├── extracing.py          # Planet class + detection helpers
├── best_model.pth        # Trained MobileNetV2 weights
├── Data/                 # Processed + augmented images
│   ├── Earth/            # Background-removed Earth images
│   ├── Moon/             # Background-removed Moon images
│   └── ...              # Other planets
├── planets/              # Source dataset (CC BY-NC 4.0)
└── real/                # Experiments with real robot camera images
```

## Tech Stack

- **Python** - All the code
- **PyTorch** - Neural network training
- **OpenCV** - Circle detection, image processing
- **torchvision** - MobileNetV2, transforms

## Running It

### Extract planets from images

```python
from extracting import planetDetection

planetDetection("planets/Planets/Earth", "Data")
planetDetection("planets/Planets/Moon", "Data")
```

### Generate augmented dataset

```python
from manipulation import main

main()  # Processes all images in Data/
```

### Train the model

Open `model.ipynb` and run through the cells.

### Make predictions

```python
from model import predict_image

predicted_class, confidence = predict_image("some_planet.jpg")
```

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 99.97% |
| Test Accuracy       | 99.95% |
| Training Samples    | 44,347 |
| Validation Samples  | 11,087 |
| Test Samples        | 6,160  |
| Epochs to converge  | 2      |

## Dataset

Planet images sourced from the [Planets and Moons Dataset](https://github.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space) by Emirhan BULUT, licensed CC BY-NC 4.0.
