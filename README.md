# Fashion-MNIST-Datsets
Fashion-MNIST Neural Network Classifier
This repository contains a Multi-Layer Perceptron (MLP) implementation built with PyTorch for classifying Fashion-MNIST clothing items into 10 categories. The project demonstrates fundamental deep learning concepts including data loading, model training, validation, and performance visualization.
📊 Performance Metrics
The model was trained for 10 epochs with Adam optimizer (lr=0.001) on 60,000 training samples. Results demonstrate effective learning on grayscale fashion images:
MetricTrainingValidationObservationsAccuracy~88-90%~87-89%Good generalization with minimal overfittingLoss~0.30~0.35Steady convergence across epochsTraining Time2-3 min (CPU)30 sec (GPU)GPU acceleration provides 4-6× speedup
Convergence Visualization
Afficher l'image
Loss and accuracy curves showing model convergence over 10 epochs
🧠 Technical Overview
Architecture: 3-layer MLP with ReLU activations
Input (784) → Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10) → Output
Key Components:

Flatten Layer: Converts 28×28 images to 784-dimensional vectors using x.view(x.size(0), -1)
Loss Function: CrossEntropyLoss (suitable for multi-class classification)
Optimizer: Adam with adaptive learning rates
Batch Size: 32 samples per iteration

Dataset: Fashion-MNIST (Zalando Research)

10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
70,000 grayscale images (60k train / 10k test)
Image dimensions: 28×28 pixels
Normalized to [-1, 1] range

🛠️ Installation & Usage
Dependencies:
bashpip install torch torchvision matplotlib
Run Training:
bashpython fashion_mnist_training.py
Expected Output:
Downloading Fashion MNIST ...
Training samples : 60000
Testing samples : 10000

Using device: cuda

Epoch [1/10]
Batch [200/1875], Loss: 0.5234, Accuracy: 81.23%

  Training   - Loss: 0.4521, Accuracy: 85.32%
  Validation - Loss: 0.4012, Accuracy: 86.15%

✅ Training Complete!
Final Test Accuracy: 87.45%
📁 Code Structure
The implementation follows standard PyTorch training pipeline:

Data Loading: torchvision.datasets.FashionMNIST + DataLoader
Model Definition: Custom MyModel(nn.Module) class
Training Loop: Forward pass → Loss → Backprop → Weight update
Validation Loop: Evaluation with torch.no_grad() (no gradient computation)
Metrics Tracking: Loss and accuracy stored per epoch
Visualization: Matplotlib plots saved as results.png

Critical Implementation Details:
python# Flatten transformation (required for MLP)
x = x.view(x.size(0), -1)  # [batch, 1, 28, 28] → [batch, 784]

# Training mode
model.train()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Validation mode (disables dropout, batch norm)
model.eval()
with torch.no_grad():
    # Forward pass only
🔧 Hyperparameter Tuning
ParameterDefaultAlternativesImpactLearning Rate0.0010.0001 - 0.01Higher = faster but unstable; Lower = slower convergenceBatch Size3216, 64, 128Smaller = more noise; Larger = more memoryHidden Units128, 64256, 512More neurons = higher capacity but risk overfittingEpochs105 - 50More epochs may improve performance but risk overfitting
📈 Results Analysis
Strengths:

✅ Fast convergence (reaches ~85% accuracy by epoch 2)
✅ Minimal train-validation gap (good generalization)
✅ Simple architecture suitable for baseline comparisons

Limitations:

⚠️ MLP cannot capture spatial patterns (no convolutional layers)
⚠️ Accuracy plateaus ~87-89% (CNN achieves 92-94%)
⚠️ Fully connected layers are parameter-heavy (784×128 = 100k params in first layer)

🚀 Next Steps

Implement CNN: Add convolutional layers to capture spatial features
Data Augmentation: Apply random rotations/flips to improve robustness
Regularization: Add dropout (p=0.5) to prevent overfitting
Advanced Optimizers: Test SGD with momentum or AdamW
Transfer Learning: Fine-tune pre-trained ResNet/VGG models

📚 References

Fashion-MNIST Dataset - Zalando Research
PyTorch Documentation
Adam Optimizer Paper - Kingma & Ba, 2014
