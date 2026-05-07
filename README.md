CS499-Fractures

This project is a CNN built on previous work to try to refine fracture identification using AI vision models.

1. Dataset

We use the FracAtlas dataset, a musculoskeletal bone fracture dataset containing 4,083 X-ray images with annotations for fracture classification.

Source: https://figshare.com/articles/dataset/The_dataset/22363012
Classes: Fractured (717 images), Non-fractured (3,366 images)

2. Models Implemented

ResNet18: Pre-trained on ImageNet, fine-tuned for binary fracture classification
Vision Transformer (ViT): Google's ViT-base-patch16-224, compared against ResNet

3. Results

ResNet18 Test Performance:
Accuracy: 88.58%
Precision: 91.39%
Recall: 95.31%
F1-Score: 93.31%

Key Finding: The model achieves 95% recall, catching nearly all actual fractures while producing 46 false positives per 613 test images.

4. Setup and Installation

Clone the repository:
git clone https://github.com/MakaelaCrookes/CS499-Fractures.git
cd CS499-Fractures

Create virtual environment:
python -m venv venv
source venv/bin/activate  (On Windows: venv\Scripts\activate)

Install dependencies:
pip install -r requirements.txt

Download dataset from Figshare and extract to data/FracAtlas/

Train ResNet18:
python src/02_train_resnet.py

Train Vision Transformer:
python src/03_train_vit.py

5. Requirements

torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
pillow>=9.0.0
tqdm>=4.64.0
seaborn>=0.11.0
transformers>=4.25.0

6. Key Implementation Details

Error Handling: SafeImageFolder class handles corrupted images without crashing
Data Splitting: 70/15/15 train/val/test split by patient to prevent leakage
Data Augmentation: Random horizontal flips and ±10° rotations for training
Learning Rate Scheduling: ReduceLROnPlateau reduces LR when validation loss plateaus

7. References

Abedeen, I., et al. "FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs." Scientific Data (2023)

Janisch, M., et al. "Pediatric radius torus fractures in x-rays—how computer vision could render lateral projections obsolete." Frontiers in Pediatrics (2022)

8. GitHub Links

Repository: https://github.com/MakaelaCrookes/CS499-Fractures

9. Authors

Jasmine Flowers
Makaela Crookes

CS 499/599: Computer Vision for Healthcare - Spring 2026
