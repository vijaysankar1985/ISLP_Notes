# BART (Bayesian Additive Regression Trees) Implementation
# This is a simplified educational implementation of the BART algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SimpleBARTRegressor:
    """
    Simplified BART (Bayesian Additive Regression Trees) implementation
    
    This implements the algorithm from the paper:
    1. Initialize K trees with simple predictions
    2. For each iteration, update each tree by fitting to partial residuals
    3. Average predictions after burn-in period
    """
    
    def __init__(self, n_trees=50, n_iterations=100, burn_in=20, max_depth=3):
        """
        Parameters:
        -----------
        n_trees : int
            Number of trees (K in the algorithm)
        n_iterations : int
            Number of iterations (B in the algorithm)
        burn_in : int
            Number of burn-in samples to discard (L in the algorithm)
        max_depth : int
            Maximum depth of individual trees
        """
        self.n_trees = n_trees  # K
        self.n_iterations = n_iterations  # B
        self.burn_in = burn_in  # L
        self.max_depth = max_depth
        
        # Storage for trees and predictions
        self.trees_history = []  # Store trees from each iteration
        self.predictions_history = []  # Store predictions from each iteration
        
    def _initialize_trees(self, X, y):
        """
        Step 1: Initialize K trees with simple predictions
        All trees start with the same prediction: mean of y divided by K
        """
        initial_prediction = np.mean(y) / self.n_trees
        trees = []
        
        for k in range(self.n_trees):
            # Create a simple tree that predicts the initial value
            tree = DecisionTreeRegressor(max_depth=1, random_state=k)
            # Fit to a simple pattern to get initial prediction
            tree.fit(X, np.full(len(y), initial_prediction))
            trees.append(tree)
            
        return trees
    
    def _compute_partial_residuals(self, X, y, trees, current_tree_idx):
        """
        Step 3a: Compute partial residuals for the current tree
        ri = yi - sum of predictions from all OTHER trees
        """
        # Get predictions from all trees except the current one
        other_predictions = np.zeros(len(y))
        
        for k, tree in enumerate(trees):
            if k != current_tree_idx:
                other_predictions += tree.predict(X)
        
        # Compute residuals
        residuals = y - other_predictions
        return residuals
    
    def _fit_tree_to_residuals(self, X, residuals, random_state):
        """
        Step 3a.ii: Fit a new tree to the partial residuals
        This simulates the "random perturbation" mentioned in the algorithm
        """
        # Create a new tree with some randomness
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=max(2, len(X) // 20),
            min_samples_leaf=max(1, len(X) // 40),
            random_state=random_state
        )
        
        # Fit to residuals
        tree.fit(X, residuals)
        return tree
    
    def fit(self, X, y):
        """
        Main BART algorithm implementation
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"Starting BART with {self.n_trees} trees, {self.n_iterations} iterations")
        print(f"Burn-in period: {self.burn_in} iterations")
        
        # Step 1: Initialize trees
        current_trees = self._initialize_trees(X, y)
        
        # Store initial state
        self.trees_history.append([tree for tree in current_trees])
        
        # Main algorithm loop
        for b in range(1, self.n_iterations + 1):
            if b % 20 == 0:
                print(f"Iteration {b}/{self.n_iterations}")
            
            # Step 3: For each tree, update it
            new_trees = []
            
            for k in range(self.n_trees):
                # Step 3a.i: Compute partial residuals
                residuals = self._compute_partial_residuals(X, y, current_trees, k)
                
                # Step 3a.ii: Fit new tree to residuals
                new_tree = self._fit_tree_to_residuals(
                    X, residuals, 
                    random_state=b * self.n_trees + k
                )
                new_trees.append(new_tree)
            
            # Update current trees
            current_trees = new_trees
            
            # Store this iteration's trees
            self.trees_history.append([tree for tree in current_trees])
            
            # Step 3b: Compute combined prediction for this iteration
            combined_pred = np.zeros(len(X))
            for tree in current_trees:
                combined_pred += tree.predict(X)
            
            self.predictions_history.append(combined_pred)
        
        print("BART training completed!")
        return self
    
    def predict(self, X):
        """
        Step 4: Compute final prediction by averaging post-burn-in samples
        """
        X = np.array(X)
        
        if len(self.predictions_history) == 0:
            raise ValueError("Model not fitted yet!")
        
        # Use only post-burn-in predictions
        post_burnin_start = self.burn_in
        if post_burnin_start >= len(self.predictions_history):
            post_burnin_start = len(self.predictions_history) // 2
        
        # Get predictions from post-burn-in iterations
        post_burnin_predictions = []
        
        for b in range(post_burnin_start, len(self.trees_history)):
            trees = self.trees_history[b]
            pred = np.zeros(len(X))
            for tree in trees:
                pred += tree.predict(X)
            post_burnin_predictions.append(pred)
        
        # Step 4: Average the predictions
        if len(post_burnin_predictions) == 0:
            # Fallback to last iteration if no post-burn-in samples
            trees = self.trees_history[-1]
            final_pred = np.zeros(len(X))
            for tree in trees:
                final_pred += tree.predict(X)
            return final_pred
        
        final_prediction = np.mean(post_burnin_predictions, axis=0)
        return final_prediction
    
    def plot_convergence(self, X_train, y_train):
        """
        Plot how the model's predictions improve over iterations
        """
        if len(self.predictions_history) == 0:
            print("No training history available")
            return
        
        # Calculate MSE for each iteration
        mse_history = []
        for pred in self.predictions_history:
            mse = mean_squared_error(y_train, pred)
            mse_history.append(mse)
        
        plt.figure(figsize=(12, 4))
        
        # Plot MSE over iterations
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(mse_history) + 1), mse_history)
        plt.axvline(x=self.burn_in, color='r', linestyle='--', 
                   label=f'Burn-in (L={self.burn_in})')
        plt.xlabel('Iteration')
        plt.ylabel('Training MSE')
        plt.title('BART Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot final predictions vs actual
        plt.subplot(1, 2, 2)
        final_pred = self.predict(X_train)
        plt.scatter(y_train, final_pred, alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('BART Predictions')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstration and Usage Example
print("="*60)
print("BART Algorithm Implementation")
print("="*60)

# Generate sample data
print("\n1. Generating sample data...")
X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create and train BART model
print("\n2. Training BART model...")
bart = SimpleBARTRegressor(
    n_trees=20,          # K = 20 trees
    n_iterations=60,     # B = 60 iterations  
    burn_in=15,          # L = 15 burn-in samples
    max_depth=3
)

# Fit the model
bart.fit(X_train, y_train)

# Make predictions
print("\n3. Making predictions...")
y_pred_train = bart.predict(X_train)
y_pred_test = bart.predict(X_test)

# Evaluate performance
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"\nResults:")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Plot convergence
print("\n4. Plotting convergence...")
bart.plot_convergence(X_train, y_train)

# Compare with simple decision tree
print("\n5. Comparison with single decision tree...")
from sklearn.tree import DecisionTreeRegressor

simple_tree = DecisionTreeRegressor(max_depth=6, random_state=42)
simple_tree.fit(X_train, y_train)
simple_pred = simple_tree.predict(X_test)
simple_mse = mean_squared_error(y_test, simple_pred)

print(f"Single Decision Tree MSE: {simple_mse:.4f}")
print(f"BART MSE: {test_mse:.4f}")
print(f"Improvement: {((simple_mse - test_mse) / simple_mse * 100):.1f}%")

print("\n" + "="*60)
print("BART Algorithm Summary:")
print("="*60)
print("1. Starts with K simple trees (all predicting mean/K)")
print("2. For each iteration:")
print("   - For each tree k:")
print("     * Compute partial residuals (what other trees miss)")
print("     * Fit new tree to capture these residuals")
print("   - Store the combined prediction")
print("3. Final prediction = average of post-burn-in predictions")
print("4. This ensemble approach captures complex patterns")
print("   that single trees cannot model effectively")
print("="*60)