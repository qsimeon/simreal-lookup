"""
Joint Gradient Descent and Linear Regression Optimization of Neural Networks
with Automatic Lookup of Similar Real Data

This demo implements a hybrid optimization approach that:
1. Uses gradient descent for neural network training
2. Incorporates linear regression for parameter refinement
3. Automatically looks up similar real data points to guide optimization
4. Uses distance metrics to find relevant real data samples

The system maintains a database of real data and uses it to augment
the training process by finding similar examples.
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataPoint:
    """Represents a single data point with features and label"""
    features: np.ndarray
    label: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealDataStore:
    """
    Stores and manages real data points for similarity lookup.
    Uses efficient distance calculations to find similar examples.
    """
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.data_points: List[DataPoint] = []
        self.features_matrix: Optional[np.ndarray] = None
        self.labels_vector: Optional[np.ndarray] = None
        
    def add_data_point(self, features: np.ndarray, label: float, metadata: Dict = None):
        """Add a real data point to the store"""
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {features.shape[0]}")
        
        data_point = DataPoint(features=features.copy(), label=label, metadata=metadata or {})
        self.data_points.append(data_point)
        self._rebuild_matrices()
        
    def add_batch(self, features_batch: np.ndarray, labels_batch: np.ndarray):
        """Add multiple data points at once"""
        if features_batch.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {features_batch.shape[1]}")
        
        for features, label in zip(features_batch, labels_batch):
            self.data_points.append(DataPoint(features=features.copy(), label=label))
        
        self._rebuild_matrices()
        
    def _rebuild_matrices(self):
        """Rebuild feature and label matrices for efficient computation"""
        if len(self.data_points) == 0:
            self.features_matrix = None
            self.labels_vector = None
            return
            
        self.features_matrix = np.vstack([dp.features for dp in self.data_points])
        self.labels_vector = np.array([dp.label for dp in self.data_points])
        
    def find_similar(self, query_features: np.ndarray, k: int = 5, 
                     distance_metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k most similar data points to the query features.
        
        Returns:
            similar_features: (k, feature_dim) array of similar features
            similar_labels: (k,) array of corresponding labels
            distances: (k,) array of distances
        """
        if self.features_matrix is None or len(self.data_points) == 0:
            raise ValueError("No data points in store")
        
        if query_features.shape[0] != self.feature_dim:
            raise ValueError(f"Query feature dimension mismatch")
        
        # Calculate distances
        if distance_metric == 'euclidean':
            distances = np.linalg.norm(self.features_matrix - query_features, axis=1)
        elif distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            query_norm = np.linalg.norm(query_features)
            data_norms = np.linalg.norm(self.features_matrix, axis=1)
            
            if query_norm == 0:
                distances = np.ones(len(self.data_points))
            else:
                cosine_sim = np.dot(self.features_matrix, query_features) / (data_norms * query_norm + 1e-8)
                distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Get k nearest neighbors
        k = min(k, len(self.data_points))
        nearest_indices = np.argpartition(distances, k-1)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        
        similar_features = self.features_matrix[nearest_indices]
        similar_labels = self.labels_vector[nearest_indices]
        similar_distances = distances[nearest_indices]
        
        return similar_features, similar_labels, similar_distances
    
    def save(self, filepath: str):
        """Save the data store to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'RealDataStore':
        """Load a data store from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class NeuralNetwork:
    """
    Simple feedforward neural network with one hidden layer.
    Supports both gradient descent and linear regression optimization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 learning_rate: float = 0.01, activation: str = 'relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Cache for forward pass
        self.cache = {}
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function"""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sig * (1 - sig)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # Input to hidden
        self.cache['X'] = X
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self._activate(self.cache['Z1'])
        
        # Hidden to output
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.cache['Z2']  # Linear output for regression
        
        return self.cache['A2']
    
    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients"""
        m = y_true.shape[0]
        
        # Output layer gradients
        dZ2 = (self.cache['A2'] - y_true) / m
        dW2 = np.dot(self.cache['A1'].T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activate_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update weights using gradients"""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute mean squared error loss"""
        return np.mean((y_pred - y_true) ** 2)
    
    def get_hidden_representation(self, X: np.ndarray) -> np.ndarray:
        """Get the hidden layer representation for input X"""
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._activate(Z1)
        return A1


class HybridOptimizer:
    """
    Hybrid optimizer that combines gradient descent with linear regression
    and uses real data lookup for guidance.
    """
    
    def __init__(self, neural_net: NeuralNetwork, real_data_store: RealDataStore,
                 similarity_weight: float = 0.3, k_neighbors: int = 5,
                 use_linear_regression: bool = True):
        self.nn = neural_net
        self.data_store = real_data_store
        self.similarity_weight = similarity_weight
        self.k_neighbors = k_neighbors
        self.use_linear_regression = use_linear_regression
        self.training_history = []
        
    def _get_augmented_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training batch with similar real data points.
        For each simulated data point, find similar real data and include them.
        """
        if len(self.data_store.data_points) == 0:
            return X_batch, y_batch
        
        augmented_X = [X_batch]
        augmented_y = [y_batch]
        
        try:
            for x_sim in X_batch:
                similar_features, similar_labels, distances = self.data_store.find_similar(
                    x_sim, k=self.k_neighbors
                )
                
                # Weight similar samples by inverse distance
                weights = 1.0 / (distances + 1e-8)
                weights = weights / np.sum(weights)
                
                # Add weighted similar samples
                for feat, label, weight in zip(similar_features, similar_labels, weights):
                    if weight > 0.1:  # Only include sufficiently similar samples
                        augmented_X.append(feat.reshape(1, -1))
                        augmented_y.append(np.array([[label]]))
        
        except ValueError:
            # If no similar data found, return original batch
            pass
        
        if len(augmented_X) > 1:
            return np.vstack(augmented_X), np.vstack(augmented_y)
        else:
            return X_batch, y_batch
    
    def _linear_regression_refinement(self, X: np.ndarray, y: np.ndarray):
        """
        Refine output layer weights using linear regression on hidden representations.
        This provides a closed-form solution for the output layer.
        """
        # Get hidden representations
        hidden = self.nn.get_hidden_representation(X)
        
        # Add bias term
        hidden_with_bias = np.hstack([hidden, np.ones((hidden.shape[0], 1))])
        
        # Solve linear regression: y = hidden_with_bias @ theta
        try:
            # Use pseudo-inverse for numerical stability
            theta = np.linalg.lstsq(hidden_with_bias, y, rcond=None)[0]
            
            # Update output layer weights
            self.nn.W2 = theta[:-1]
            self.nn.b2 = theta[-1:]
            
        except np.linalg.LinAlgError:
            # If linear regression fails, skip this step
            pass
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   use_augmentation: bool = True) -> float:
        """
        Perform one training step with hybrid optimization.
        
        Steps:
        1. Augment batch with similar real data
        2. Forward pass
        3. Backward pass and gradient descent
        4. Optional linear regression refinement
        """
        # Augment batch with similar real data
        if use_augmentation:
            X_augmented, y_augmented = self._get_augmented_batch(X_batch, y_batch)
        else:
            X_augmented, y_augmented = X_batch, y_batch
        
        # Forward pass
        y_pred = self.nn.forward(X_augmented)
        loss = self.nn.compute_loss(y_pred, y_augmented)
        
        # Backward pass and weight update (gradient descent)
        gradients = self.nn.backward(y_augmented)
        self.nn.update_weights(gradients)
        
        # Linear regression refinement (every few steps)
        if self.use_linear_regression and len(self.training_history) % 10 == 0:
            self._linear_regression_refinement(X_augmented, y_augmented)
        
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 100, batch_size: int = 32, 
              verbose: bool = True) -> List[float]:
        """
        Train the neural network using hybrid optimization.
        """
        n_samples = X_train.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.nn.forward(X)


def generate_synthetic_data(n_samples: int, feature_dim: int, 
                           noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for demonstration.
    y = sum(x_i^2) + noise
    """
    X = np.random.randn(n_samples, feature_dim)
    y = np.sum(X ** 2, axis=1, keepdims=True) + noise_level * np.random.randn(n_samples, 1)
    return X, y


def generate_real_data(n_samples: int, feature_dim: int, 
                       noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 'real' data (similar function but with different characteristics).
    This simulates having a database of real measurements.
    """
    X = np.random.randn(n_samples, feature_dim) * 0.8  # Slightly different distribution
    y = np.sum(X ** 2, axis=1, keepdims=True) + noise_level * np.random.randn(n_samples, 1)
    return X, y


def demo_basic_usage():
    """Demonstrate basic usage of the hybrid optimizer"""
    print("=" * 80)
    print("DEMO 1: Basic Hybrid Optimization")
    print("=" * 80)
    
    # Configuration
    feature_dim = 5
    hidden_dim = 10
    n_real_samples = 200
    n_train_samples = 100
    n_test_samples = 50
    
    print(f"\nConfiguration:")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Hidden layer size: {hidden_dim}")
    print(f"  Real data samples: {n_real_samples}")
    print(f"  Training samples: {n_train_samples}")
    print(f"  Test samples: {n_test_samples}")
    
    # Create real data store and populate it
    print("\n[1] Creating real data store...")
    data_store = RealDataStore(feature_dim=feature_dim)
    X_real, y_real = generate_real_data(n_real_samples, feature_dim)
    data_store.add_batch(X_real, y_real.flatten())
    print(f"    Added {len(data_store.data_points)} real data points")
    
    # Generate training and test data
    print("\n[2] Generating training and test data...")
    X_train, y_train = generate_synthetic_data(n_train_samples, feature_dim)
    X_test, y_test = generate_synthetic_data(n_test_samples, feature_dim)
    print(f"    Training set: {X_train.shape}")
    print(f"    Test set: {X_test.shape}")
    
    # Create neural network
    print("\n[3] Creating neural network...")
    nn = NeuralNetwork(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        learning_rate=0.01,
        activation='relu'
    )
    print(f"    Network: {feature_dim} -> {hidden_dim} -> 1")
    
    # Create hybrid optimizer
    print("\n[4] Creating hybrid optimizer...")
    optimizer = HybridOptimizer(
        neural_net=nn,
        real_data_store=data_store,
        similarity_weight=0.3,
        k_neighbors=5,
        use_linear_regression=True
    )
    print("    Optimizer configured with:")
    print(f"      - Similarity weight: 0.3")
    print(f"      - K neighbors: 5")
    print(f"      - Linear regression: Enabled")
    
    # Train the model
    print("\n[5] Training model...")
    losses = optimizer.train(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n[6] Evaluating on test set...")
    y_pred = optimizer.predict(X_test)
    test_loss = np.mean((y_pred - y_test) ** 2)
    print(f"    Test MSE: {test_loss:.6f}")
    
    # Show some predictions
    print("\n[7] Sample predictions:")
    for i in range(min(5, len(X_test))):
        print(f"    Sample {i+1}: True={y_test[i,0]:.4f}, Pred={y_pred[i,0]:.4f}, "
              f"Error={abs(y_test[i,0]-y_pred[i,0]):.4f}")
    
    return optimizer, losses, test_loss


def demo_similarity_lookup():
    """Demonstrate the similarity lookup functionality"""
    print("\n" + "=" * 80)
    print("DEMO 2: Similarity Lookup Demonstration")
    print("=" * 80)
    
    feature_dim = 3
    
    # Create data store with known data
    print("\n[1] Creating data store with known points...")
    data_store = RealDataStore(feature_dim=feature_dim)
    
    # Add some specific data points
    known_points = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 2.0),
        (np.array([0.0, 0.0, 1.0]), 3.0),
        (np.array([1.0, 1.0, 0.0]), 4.0),
        (np.array([0.5, 0.5, 0.5]), 5.0),
    ]
    
    for features, label in known_points:
        data_store.add_data_point(features, label)
        print(f"    Added: features={features}, label={label}")
    
    # Query with a test point
    print("\n[2] Querying similar points...")
    query = np.array([0.9, 0.1, 0.0])
    print(f"    Query point: {query}")
    
    similar_features, similar_labels, distances = data_store.find_similar(
        query, k=3, distance_metric='euclidean'
    )
    
    print("\n    Top 3 similar points:")
    for i, (feat, label, dist) in enumerate(zip(similar_features, similar_labels, distances)):
        print(f"      {i+1}. Features={feat}, Label={label:.1f}, Distance={dist:.4f}")
    
    # Try cosine distance
    print("\n[3] Querying with cosine distance...")
    similar_features, similar_labels, distances = data_store.find_similar(
        query, k=3, distance_metric='cosine'
    )
    
    print("    Top 3 similar points (cosine):")
    for i, (feat, label, dist) in enumerate(zip(similar_features, similar_labels, distances)):
        print(f"      {i+1}. Features={feat}, Label={label:.1f}, Distance={dist:.4f}")


def demo_comparison():
    """Compare hybrid optimizer with standard gradient descent"""
    print("\n" + "=" * 80)
    print("DEMO 3: Comparison with Standard Gradient Descent")
    print("=" * 80)
    
    feature_dim = 4
    hidden_dim = 8
    n_real_samples = 150
    n_train_samples = 80
    n_test_samples = 40
    epochs = 30
    
    print(f"\nConfiguration: {feature_dim}D input, {hidden_dim} hidden units, {epochs} epochs")
    
    # Prepare data
    print("\n[1] Preparing data...")
    data_store = RealDataStore(feature_dim=feature_dim)
    X_real, y_real = generate_real_data(n_real_samples, feature_dim)
    data_store.add_batch(X_real, y_real.flatten())
    
    X_train, y_train = generate_synthetic_data(n_train_samples, feature_dim)
    X_test, y_test = generate_synthetic_data(n_test_samples, feature_dim)
    
    # Train with hybrid optimizer
    print("\n[2] Training with HYBRID optimizer...")
    nn_hybrid = NeuralNetwork(feature_dim, hidden_dim, 1, learning_rate=0.01)
    optimizer_hybrid = HybridOptimizer(
        nn_hybrid, data_store, 
        similarity_weight=0.3, 
        k_neighbors=5,
        use_linear_regression=True
    )
    losses_hybrid = optimizer_hybrid.train(X_train, y_train, epochs=epochs, verbose=False)
    y_pred_hybrid = optimizer_hybrid.predict(X_test)
    test_loss_hybrid = np.mean((y_pred_hybrid - y_test) ** 2)
    
    # Train with standard gradient descent (no augmentation, no linear regression)
    print("\n[3] Training with STANDARD gradient descent...")
    nn_standard = NeuralNetwork(feature_dim, hidden_dim, 1, learning_rate=0.01)
    optimizer_standard = HybridOptimizer(
        nn_standard, data_store,
        similarity_weight=0.0,
        k_neighbors=0,
        use_linear_regression=False
    )
    losses_standard = optimizer_standard.train(
        X_train, y_train, epochs=epochs, verbose=False, 
    )
    # Manually train without augmentation
    losses_standard = []
    for epoch in range(epochs):
        y_pred = nn_standard.forward(X_train)
        loss = nn_standard.compute_loss(y_pred, y_train)
        gradients = nn_standard.backward(y_train)
        nn_standard.update_weights(gradients)
        losses_standard.append(loss)
    
    y_pred_standard = nn_standard.forward(X_test)
    test_loss_standard = np.mean((y_pred_standard - y_test) ** 2)
    
    # Compare results
    print("\n[4] Results comparison:")
    print(f"    Hybrid Optimizer:")
    print(f"      Final training loss: {losses_hybrid[-1]:.6f}")
    print(f"      Test MSE: {test_loss_hybrid:.6f}")
    print(f"\n    Standard Gradient Descent:")
    print(f"      Final training loss: {losses_standard[-1]:.6f}")
    print(f"      Test MSE: {test_loss_standard:.6f}")
    
    improvement = ((test_loss_standard - test_loss_hybrid) / test_loss_standard) * 100
    print(f"\n    Improvement: {improvement:.2f}%")
    
    if test_loss_hybrid < test_loss_standard:
        print("    ✓ Hybrid optimizer performed BETTER!")
    else:
        print("    ✗ Standard optimizer performed better (may need tuning)")


def demo_save_load():
    """Demonstrate saving and loading data store"""
    print("\n" + "=" * 80)
    print("DEMO 4: Save and Load Data Store")
    print("=" * 80)
    
    feature_dim = 3
    filepath = "real_data_store.pkl"
    
    # Create and save
    print("\n[1] Creating and saving data store...")
    data_store = RealDataStore(feature_dim=feature_dim)
    X_real, y_real = generate_real_data(50, feature_dim)
    data_store.add_batch(X_real, y_real.flatten())
    data_store.save(filepath)
    print(f"    Saved {len(data_store.data_points)} points to '{filepath}'")
    
    # Load
    print("\n[2] Loading data store...")
    loaded_store = RealDataStore.load(filepath)
    print(f"    Loaded {len(loaded_store.data_points)} points from '{filepath}'")
    
    # Verify
    print("\n[3] Verifying data integrity...")
    query = X_real[0]
    orig_similar, orig_labels, orig_dist = data_store.find_similar(query, k=3)
    load_similar, load_labels, load_dist = loaded_store.find_similar(query, k=3)
    
    if np.allclose(orig_similar, load_similar) and np.allclose(orig_labels, load_labels):
        print("    ✓ Data integrity verified!")
    else:
        print("    ✗ Data mismatch detected!")
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"\n[4] Cleaned up '{filepath}'")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("HYBRID NEURAL NETWORK OPTIMIZER DEMONSTRATION")
    print("Joint Gradient Descent + Linear Regression + Real Data Lookup")
    print("=" * 80)
    
    try:
        # Demo 1: Basic usage
        optimizer, losses, test_loss = demo_basic_usage()
        
        # Demo 2: Similarity lookup
        demo_similarity_lookup()
        
        # Demo 3: Comparison
        demo_comparison()
        
        # Demo 4: Save/Load
        demo_save_load()
        
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("  ✓ Real data store with similarity lookup")
        print("  ✓ Hybrid optimization (gradient descent + linear regression)")
        print("  ✓ Automatic augmentation with similar real data")
        print("  ✓ Distance-based data retrieval (Euclidean & Cosine)")
        print("  ✓ Comparison with standard gradient descent")
        print("  ✓ Save/Load functionality")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
