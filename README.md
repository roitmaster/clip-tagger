# CLIP-based Image Tagging Model

This repository contains code for a CLIP-based image tagging model that can generate tags for images using a dictionary of predefined tags. This project leverages OpenAI's [CLIP](https://github.com/openai/CLIP) model to create and assign relevant tags to images based on their similarity to text embeddings.

you can look in the [notebook](Colab_notebook.ipynb) for example of use

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Tagger Class](#tagger-class)
  - [Optimization Function](#optimization-function)
  - [Creating Labels](#creating-labels)
- [Example](#example)
- [License](#license)

## Installation

To use this code, you need to have Python installed along with the following packages:
- `torch`
- `clip` (install via [OpenAI CLIP](https://github.com/openai/CLIP))
- `PIL` (for image handling)

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
### Dataset Preparation
The code includes a custom dataset class, MyDataset, that facilitates the handling of images and optional labels. This class can preprocess images for the CLIP model and is used by the Tagger class to create data loaders.
```
from PIL import Image
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    ...

```

Tagger Class
The Tagger class is responsible for:

- Loading and initializing the CLIP model and its preprocessing function.
- Generating text embeddings for each tag using CLIP.
- Creating a DataLoader to batch images for processing.
- Calculating similarity between image and text embeddings to assign relevant tags.

```
class Tagger:
    ...
```

#### Methods
- get_text_features(tag_dict): Converts tags into text embeddings.
- create_dataloader(images, labels): Creates a DataLoader for images.
- __call__(np_images, threshold, max_only): Performs tagging on input images and returns tags above a specified threshold.
  
### Optimization Function
The optimize_model function optimizes the text embeddings in the Tagger class by adjusting parameters based on image-label similarity. This can help fine-tune the tags for better accuracy.
```
def optimize_model(tagger, images, labels, lr=0.01, epochs=5, print_ever=1):
    ...
```

### Creating Labels
The create_labels function uses the Tagger to generate tags above a set score threshold, filtering out low-confidence tags to reduce noise.

```
def create_labels(tagger, images, threshold=0.6):
    ...
```
Example
1. Initialize the Tagger class with a dictionary of tags.
2. Use the call of the instance of the Tagger class to generate labels for your images.

```
tag_dict = {
    "colors": ["red", "blue", "green"],
    "objects": ["car", "tree", "dog"]
}

tagger = Tagger(tag_dict)
images = [image1, image2]  # Replace with your images in numpy array format

labels = tagger(images, threshold=0.6)
print(labels)

```


