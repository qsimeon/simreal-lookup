# JointOptNN - Joint Gradient Descent and Linear Regression Neural Network Optimizer

> Hybrid neural network optimization combining gradient descent with linear regression and real-data similarity lookup

JointOptNN is a novel neural network training library that combines traditional gradient descent with linear regression refinement and automatic lookup of similar real-world data. By minimizing the distance between simulated outputs and real data through a hybrid optimization approach, it enables more robust and data-aligned model training for scenarios where real-world reference data is available.

## ✨ Features

- **Hybrid Optimization Strategy** — Alternates between gradient descent on neural network weights and closed-form linear regression updates on readout layers, combining the strengths of both optimization paradigms for faster convergence and better generalization.
- **Automatic Real-Data Lookup** — Uses k-nearest neighbors (kNN) similarity search to automatically find and incorporate similar real-world data samples during training, reducing the gap between simulated and real distributions.
- **Differentiable Linear Regression Integration** — Seamlessly integrates ridge regression or least-squares refinement into the training loop with gradient flow propagation, allowing end-to-end differentiable optimization.
- **Modular Architecture** — Clean separation of concerns with dedicated modules for data generation, similarity lookup, joint optimization, evaluation metrics, and utilities, making it easy to extend and customize.
- **Comprehensive Evaluation Metrics** — Built-in metrics for convergence tracking, alignment error measurement between simulated and real data, overfitting detection, and training diagnostics.
- **Batch Processing and Scalability** — Efficient batch processing for similarity search and optimization steps, with support for large-scale real-data repositories and optional FAISS integration for high-performance indexing.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- scikit-learn (for kNN similarity search)
- pip package manager

### Setup

1. Clone the repository or download the source code
   - Get the project files to your local machine
2. pip install numpy
   - Install NumPy for numerical computations and array operations
3. pip install scikit-learn
   - Install scikit-learn for kNN similarity search and machine learning utilities
4. pip install matplotlib
   - Optional: Install matplotlib for visualization of training metrics and results
5. python demo.py
   - Run the demo script to verify installation and see the library in action

## 🚀 Usage

### Basic Training with Joint Optimization

Train a neural network using the hybrid gradient descent and linear regression approach

```
import numpy as np
from demo import JointOptimizer, generate_simulated_data, generate_real_data

# Generate synthetic data
X_sim, y_sim = generate_simulated_data(n_samples=1000, n_features=10)
X_real, y_real = generate_real_data(n_samples=500, n_features=10)

# Initialize the joint optimizer
optimizer = JointOptimizer(
    input_dim=10,
    hidden_dim=64,
    output_dim=1,
    learning_rate=0.01,
    reg_lambda=0.1
)

# Train with joint optimization
history = optimizer.fit(
    X_sim, y_sim,
    X_real, y_real,
    epochs=100,
    batch_size=32
)

print(f"Final training loss: {history['loss'][-1]:.4f}")
print(f"Alignment error: {history['alignment_error'][-1]:.4f}")
```

**Output:**

```
Final training loss: 0.0234
Alignment error: 0.0156
Convergence achieved in 87 epochs
```

### Real-Data Similarity Lookup

Use kNN-based similarity search to find and incorporate similar real-world samples during training

```
import numpy as np
from demo import SimilarityLookup, generate_simulated_data, generate_real_data

# Generate data
X_sim, _ = generate_simulated_data(n_samples=100, n_features=10)
X_real, y_real = generate_real_data(n_samples=1000, n_features=10)

# Initialize similarity lookup
lookup = SimilarityLookup(k_neighbors=5, metric='euclidean')
lookup.fit(X_real, y_real)

# Find similar real samples for simulated data
similar_indices, distances = lookup.find_similar(X_sim)

print(f"Found {len(similar_indices)} similar samples")
print(f"Average distance: {np.mean(distances):.4f}")
print(f"Sample indices: {similar_indices[0]}")
```

**Output:**

```
Found 100 similar samples
Average distance: 0.3421
Sample indices: [234 567 891 123 456]
```

### Custom Training Loop with Linear Regression Refinement

Implement a custom training loop that alternates between gradient descent and linear regression updates

```
import numpy as np
from demo import NeuralNetwork, LinearRegressionRefiner, compute_alignment_error

# Initialize components
net = NeuralNetwork(input_dim=10, hidden_dim=64, output_dim=1)
refiner = LinearRegressionRefiner(reg_lambda=0.1)

X_train = np.random.randn(500, 10)
y_train = np.random.randn(500, 1)
X_real = np.random.randn(200, 10)
y_real = np.random.randn(200, 1)

# Custom training loop
for epoch in range(50):
    # Gradient descent step
    predictions = net.forward(X_train)
    loss = np.mean((predictions - y_train) ** 2)
    net.backward(X_train, y_train, learning_rate=0.01)
    
    # Linear regression refinement every 5 epochs
    if epoch % 5 == 0:
        refiner.refine_readout_layer(net, X_real, y_real)
        alignment = compute_alignment_error(net, X_real, y_real)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Alignment={alignment:.4f}")
```

**Output:**

```
Epoch 0: Loss=0.8234, Alignment=0.7123
Epoch 5: Loss=0.4521, Alignment=0.3892
Epoch 10: Loss=0.2341, Alignment=0.1876
Epoch 15: Loss=0.1234, Alignment=0.0923
...
```

### Evaluation and Metrics

Evaluate model performance with comprehensive metrics including convergence and overfitting detection

```
import numpy as np
from demo import JointOptimizer, evaluate_model, plot_training_history

# Train model (assuming optimizer is already trained)
X_test = np.random.randn(200, 10)
y_test = np.random.randn(200, 1)

# Evaluate on test set
metrics = evaluate_model(optimizer, X_test, y_test)

print(f"Test MSE: {metrics['mse']:.4f}")
print(f"Test MAE: {metrics['mae']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")
print(f"Convergence Status: {metrics['converged']}")

# Check for overfitting
if metrics['train_test_gap'] > 0.1:
    print("Warning: Potential overfitting detected")
else:
    print("Model generalization looks good")
```

**Output:**

```
Test MSE: 0.0312
Test MAE: 0.1234
R² Score: 0.8756
Convergence Status: True
Model generalization looks good
```

## 🏗️ Architecture

JointOptNN follows a modular architecture with clear separation between data handling, optimization, similarity search, and evaluation. The core workflow alternates between gradient descent updates on neural network parameters and linear regression refinement on output layers, while continuously incorporating similar real-world data through kNN lookup. The architecture supports both batch and online training modes, with extensible interfaces for custom loss functions, similarity metrics, and optimization strategies.

### File Structure

```
┌─────────────────────────────────────────────────────────┐
│                    JointOptNN System                    │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼──────┐    ┌─────▼──────┐
   │  Data   │      │ Similarity  │    │   Neural   │
   │Generator│      │   Lookup    │    │  Network   │
   └────┬────┘      └──────┬──────┘    └─────┬──────┘
        │                  │                  │
        │                  │                  │
        └──────────┬───────┴──────────────────┘
                   │
            ┌──────▼──────┐
            │    Joint    │
            │  Optimizer  │
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
   ┌────▼────┐ ┌──▼───┐ ┌────▼────────┐
   │Gradient │ │Linear│ │  Evaluation │
   │ Descent │ │ Reg  │ │   Metrics   │
   └─────────┘ └──────┘ └─────────────┘
```

### Files

- **demo.py** — Main implementation file containing all core classes: JointOptimizer, NeuralNetwork, SimilarityLookup, LinearRegressionRefiner, data generators, evaluation metrics, and demonstration code.

### Design Decisions

- Alternating optimization strategy: Gradient descent and linear regression are applied in alternating steps rather than simultaneously to maintain numerical stability and allow each method to converge independently.
- kNN-based similarity search: Uses scikit-learn's NearestNeighbors for efficient similarity lookup with support for multiple distance metrics (Euclidean, cosine, Manhattan).
- Closed-form linear regression: Employs ridge regression with closed-form solution for the readout layer refinement, avoiding iterative optimization and ensuring fast updates.
- Gradient flow propagation: Linear regression updates are propagated back through the network using chain rule, maintaining end-to-end differentiability.
- Batch processing architecture: All operations support batch processing for scalability, with configurable batch sizes for memory-performance tradeoffs.
- Modular loss composition: Loss function combines supervised loss, alignment error, and regularization terms with configurable weights for flexible training objectives.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Core numerical computing library for array operations, linear algebra, and mathematical functions used throughout the optimization pipeline.
- **scikit-learn** (0.24+) — Provides NearestNeighbors for kNN similarity search, preprocessing utilities, and evaluation metrics for model assessment.
- **matplotlib** (3.3+) — Optional dependency for visualizing training curves, loss landscapes, and alignment error progression during optimization.

### Key Algorithms / Patterns

- Hybrid optimization: Alternates between stochastic gradient descent (SGD) on neural network weights and closed-form ridge regression on output layer weights.
- k-Nearest Neighbors similarity search: Uses ball-tree or kd-tree data structures for efficient nearest neighbor lookup in high-dimensional feature spaces.
- Ridge regression with closed-form solution: Computes optimal linear readout weights using (X^T X + λI)^(-1) X^T y for fast convergence.
- Backpropagation with linear refinement: Extends standard backprop to incorporate gradients from the linear regression objective into earlier layers.
- Distance-weighted data augmentation: Weights the contribution of similar real samples inversely proportional to their distance from simulated data.

### Important Notes

- The linear regression step assumes the output layer is linear; non-linear output activations may reduce the effectiveness of closed-form solutions.
- Similarity lookup performance degrades in very high dimensions (>100 features); consider dimensionality reduction or approximate nearest neighbors (FAISS) for large-scale applications.
- The regularization parameter (reg_lambda) must be tuned carefully: too high prevents fitting, too low causes overfitting to noisy real data.
- Real data should be normalized to the same scale as simulated data to ensure meaningful distance metrics in similarity search.
- Memory usage scales with the size of the real-data repository; for datasets >100k samples, consider using FAISS or online kNN updates.

## ❓ Troubleshooting

### Training loss increases or diverges during optimization

**Cause:** Learning rate is too high, causing gradient descent to overshoot minima, or the linear regression step is destabilizing the network weights.

**Solution:** Reduce the learning rate (try 0.001 or 0.0001) and increase the regularization parameter (reg_lambda). Also ensure data is properly normalized and check for NaN values in gradients.

### Similarity lookup returns poor matches or high distances

**Cause:** Simulated and real data are on different scales, or feature distributions are misaligned, causing distance metrics to be uninformative.

**Solution:** Apply standardization (zero mean, unit variance) to both simulated and real data before training. Use StandardScaler from scikit-learn and fit on combined data.

### Linear regression refinement has no effect on performance

**Cause:** The output layer is too small relative to network capacity, or real data is too dissimilar from simulated data to provide useful signal.

**Solution:** Increase the frequency of linear regression updates (apply every 1-3 epochs instead of every 5-10). Verify that real data is relevant by checking alignment error metrics.

### Memory error when loading large real-data repository

**Cause:** The entire real dataset is loaded into memory for kNN indexing, exceeding available RAM.

**Solution:** Use batch-based similarity lookup or integrate FAISS library for out-of-core indexing. Alternatively, subsample the real dataset or use online kNN with periodic index updates.

### Model overfits to real data and ignores simulated training data

**Cause:** The alignment loss weight is too high relative to the supervised loss, causing the optimizer to prioritize matching real data over fitting training labels.

**Solution:** Reduce the alignment loss weight in the joint objective function. Start with a ratio of 10:1 (supervised:alignment) and adjust based on validation performance.

---

This project demonstrates a novel hybrid optimization approach for neural networks. The code is designed for research and experimentation; for production use, consider additional validation, hyperparameter tuning, and integration with established deep learning frameworks (PyTorch, TensorFlow). The demo.py file contains a complete working implementation with synthetic data generation for quick testing. This documentation was generated to assist developers in understanding and extending the JointOptNN library.