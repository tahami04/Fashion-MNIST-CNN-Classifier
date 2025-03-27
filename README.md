# Fashion-MNIST-CNN-Classifier
A convolutional neural network (CNN) implementation for classifying Fashion MNIST images. Includes data preprocessing, model architecture with LeakyReLU, training/validation analysis, and visualization of results.
# Fashion MNIST Classification with CNN

## Project Overview
This repository contains a CNN model built with Keras/TensorFlow to classify Fashion MNIST images into 10 categories (e.g., T-shirts, bags). The project focuses on:
- CNN architecture design with LeakyReLU activations
- Training/validation performance analysis
- Overfitting diagnosis through accuracy/loss plots
- Final test accuracy of **91.23%**

### Key Features:
- Image preprocessing and one-hot encoding
- Deep CNN with 3 convolutional blocks
- Training visualization (accuracy/loss curves)
- Model evaluation on test data

## Repository Structure
- `fashion_mnist_cnn.ipynb`: Main Python script
- `README.md`: This documentation

## Code Overview

### 1. Data Preparation
- Loads Fashion MNIST dataset (60k training, 10k test images)
- Reshapes images to (28, 28, 1) and normalizes to [0, 1]
- One-hot encodes labels (e.g., `9` → `[0 0 ... 1]`)
- Splits training data into 48k training / 12k validation sets

### 2. CNN Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='linear', padding='same'), LeakyReLU(0.1), MaxPooling2D(),
    Conv2D(64, (3,3), activation='linear', padding='same'), LeakyReLU(0.1), MaxPooling2D(),
    Conv2D(128, (3,3), activation='linear', padding='same'), LeakyReLU(0.1), MaxPooling2D(),
    Flatten(),
    Dense(128), LeakyReLU(0.1),
    Dense(10, activation='softmax')
])
Total Parameters: 356,234

Uses LeakyReLU (α=0.1) instead of standard ReLU

Adam optimizer with categorical crossentropy

3. Training & Results
20 Epochs with batch size 64

Final Metrics:

Training Accuracy: 98.84%

Validation Accuracy: 91.76%

Test Accuracy: 91.23%

Overfitting Observed:

Training loss ↓ 0.03 vs Validation loss ↑ 0.46 (Epoch 20)

Gap between training/validation accuracy after epoch 5

How to Use
Install Dependencies:

bash
Copy
pip install tensorflow numpy matplotlib scikit-learn
Run the Script:

bash
Copy
python fashion_mnist_cnn.ipynb
Will display sample images, model summary, and training plots

Results Interpretation
Sample Visualization: Shows first training/test images (class 9 = ankle boot)

Performance Plots:

Training accuracy ↑ to 98.8% while validation plateaus at ~92%

Validation loss ↗ after epoch 5 indicates overfitting

Test Performance: Model generalizes well (91.23% accuracy) despite overfitting

Future Improvements
Address Overfitting:

Add Dropout/BatchNormalization layers

Implement data augmentation (rotation/flipping)

Model Optimization:

Experiment with different kernels/filters

Try pre-trained models (transfer learning)

Deployment:

Convert to TensorFlow Lite for mobile deployment

Build web interface with Gradio/Streamlit
