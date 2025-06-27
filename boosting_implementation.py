# Boosting Algorithm Implementation
# AdaBoost and Gradient Boosting from scratch with detailed explanations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class SimpleAdaBoostClassifier:
    """
    Simple AdaBoost (Adaptive Boosting) implementation for classification
    
    Algorithm:
    1. Start with equal weights for all samples
    2. For each iteration:
       - Train weak learner on weighted data
       - Calculate error and learner weight
       - Update sample weights (increase for misclassified)
    3. Final prediction = weighted vote of all learners
    """
    
    def __init__(self, n_estimators=50, max_depth=1):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of weak learners (boosting rounds)
        max_depth : int
            Maximum depth of decision stumps (weak learners)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
    def fit(self, X, y):
        """
        Train AdaBoost classifier
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        # Step 1: Initialize weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        print(f"Training AdaBoost with {self.n_estimators} estimators...")
        
        for t in range(self.n_estimators):
            if (t + 1) % 10 == 0:
                print(f"  Round {t + 1}/{self.n_estimators}")
            
            # Step 2a: Train weak learner on weighted data
            weak_learner = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=t
            )
            weak_learner.fit(X, y, sample_weight=sample_weights)
            
            # Step 2b: Make predictions
            predictions = weak_learner.predict(X)
            
            # Step 2c: Calculate weighted error
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Avoid division by zero and ensure error < 0.5
            error = max(error, 1e-10)
            if error >= 0.5:
                if len(self.estimators) == 0:
                    # If first estimator is bad, use it anyway with small weight
                    error = 0.4999
                else:
                    # Stop if error is too high
                    break
            
            # Step 2d: Calculate estimator weight (alpha)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Step 2e: Update sample weights
            sample_weights = sample_weights * np.exp(-alpha * y * predictions)
            sample_weights = sample_weights / np.sum(sample_weights)  # Normalize
            
            # Store the weak learner and its weight
            self.estimators.append(weak_learner)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)
        
        print(f"AdaBoost training completed with {len(self.estimators)} estimators!")
        return self
    
    def predict(self, X):
        """
        Make predictions using weighted majority vote
        """
        X = np.array(X)
        
        if len(self.estimators) == 0:
            raise ValueError("Model not fitted yet!")
        
        # Get predictions from all weak learners
        predictions = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            pred = estimator.predict(X)
            predictions += weight * pred
        
        # Return sign of weighted sum (majority vote)
        return np.sign(predictions).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        X = np.array(X)
        
        # Get weighted predictions
        weighted_sum = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            pred = estimator.predict(X)
            weighted_sum += weight * pred
        
        # Convert to probabilities using sigmoid-like function
        proba = 1 / (1 + np.exp(-2 * weighted_sum))
        
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])
    
    def plot_learning_curve(self, X_train, y_train, X_test, y_test):
        """
        Plot how error decreases with more estimators
        """
        train_errors = []
        test_errors = []
        
        for i in range(1, len(self.estimators) + 1):
            # Temporarily reduce estimators
            temp_estimators = self.estimators[:i]
            temp_weights = self.estimator_weights[:i]
            
            # Calculate predictions
            train_pred = np.zeros(len(X_train))
            test_pred = np.zeros(len(X_test))
            
            for est, weight in zip(temp_estimators, temp_weights):
                train_pred += weight * est.predict(X_train)
                test_pred += weight * est.predict(X_test)
            
            train_pred = np.sign(train_pred).astype(int)
            test_pred = np.sign(test_pred).astype(int)
            
            train_error = 1 - accuracy_score(y_train, train_pred)
            test_error = 1 - accuracy_score(y_test, test_pred)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
        
        plt.figure(figsize=(12, 4))
        
        # Plot error curves
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_errors) + 1), train_errors, 'b-', label='Training Error')
        plt.plot(range(1, len(test_errors) + 1), test_errors, 'r-', label='Test Error')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Error Rate')
        plt.title('AdaBoost Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot estimator weights
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(self.estimator_weights) + 1), self.estimator_weights)
        plt.xlabel('Estimator Index')
        plt.ylabel('Weight (Alpha)')
        plt.title('Estimator Weights')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class SimpleGradientBoostingRegressor:
    """
    Simple Gradient Boosting implementation for regression
    
    Algorithm:
    1. Start with initial prediction (mean of y)
    2. For each iteration:
       - Calculate residuals (negative gradients)
       - Train weak learner on residuals
       - Update predictions with learning rate
    3. Final prediction = initial + sum of all weak learner predictions
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        learning_rate : float
            Step size for each weak learner contribution
        max_depth : int
            Maximum depth of regression trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators = []
        self.initial_prediction = None
        self.train_scores = []
        
    def fit(self, X, y):
        """
        Train Gradient Boosting regressor
        """
        X = np.array(X)
        y = np.array(y)
        
        # Step 1: Initialize with mean prediction
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(len(y), self.initial_prediction)
        
        print(f"Training Gradient Boosting with {self.n_estimators} estimators...")
        print(f"Learning rate: {self.learning_rate}")
        
        for t in range(self.n_estimators):
            if (t + 1) % 20 == 0:
                print(f"  Round {t + 1}/{self.n_estimators}")
            
            # Step 2a: Calculate residuals (negative gradients)
            residuals = y - current_predictions
            
            # Step 2b: Train weak learner on residuals
            weak_learner = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=t
            )
            weak_learner.fit(X, residuals)
            
            # Step 2c: Make predictions and update
            weak_predictions = weak_learner.predict(X)
            current_predictions += self.learning_rate * weak_predictions
            
            # Store the weak learner and training score
            self.estimators.append(weak_learner)
            mse = mean_squared_error(y, current_predictions)
            self.train_scores.append(mse)
        
        print("Gradient Boosting training completed!")
        return self
    
    def predict(self, X):
        """
        Make predictions using all weak learners
        """
        X = np.array(X)
        
        if self.initial_prediction is None:
            raise ValueError("Model not fitted yet!")
        
        # Start with initial prediction
        predictions = np.full(len(X), self.initial_prediction)
        
        # Add contributions from all weak learners
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
        
        return predictions
    
    def plot_learning_curve(self, X_train, y_train, X_test, y_test):
        """
        Plot how MSE decreases with more estimators
        """
        train_scores = []
        test_scores = []
        
        # Calculate scores for each number of estimators
        train_pred = np.full(len(X_train), self.initial_prediction)
        test_pred = np.full(len(X_test), self.initial_prediction)
        
        train_scores.append(mean_squared_error(y_train, train_pred))
        test_scores.append(mean_squared_error(y_test, test_pred))
        
        for i, estimator in enumerate(self.estimators):
            train_pred += self.learning_rate * estimator.predict(X_train)
            test_pred += self.learning_rate * estimator.predict(X_test)
            
            train_scores.append(mean_squared_error(y_train, train_pred))
            test_scores.append(mean_squared_error(y_test, test_pred))
        
        plt.figure(figsize=(12, 4))
        
        # Plot MSE curves
        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_scores)), train_scores, 'b-', label='Training MSE')
        plt.plot(range(len(test_scores)), test_scores, 'r-', label='Test MSE')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Mean Squared Error')
        plt.title('Gradient Boosting Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot actual vs predicted
        plt.subplot(1, 2, 2)
        final_pred = self.predict(X_train)
        plt.scatter(y_train, final_pred, alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstration and Usage Examples
print("="*70)
print("BOOSTING ALGORITHMS IMPLEMENTATION")
print("="*70)

# Example 1: AdaBoost for Classification
print("\n" + "="*50)
print("1. ADABOOST CLASSIFICATION EXAMPLE")
print("="*50)

# Generate classification data
print("\nGenerating classification data...")
X_class, y_class = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                      n_redundant=2, n_clusters_per_class=1, 
                                      random_state=42)
# Convert to binary classification (-1, 1)
y_class = np.where(y_class == 0, -1, 1)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42)

print(f"Training data shape: {X_train_class.shape}")
print(f"Test data shape: {X_test_class.shape}")

# Train AdaBoost
print("\nTraining AdaBoost classifier...")
ada_boost = SimpleAdaBoostClassifier(n_estimators=30, max_depth=1)
ada_boost.fit(X_train_class, y_train_class)

# Make predictions
y_pred_ada = ada_boost.predict(X_test_class)
ada_accuracy = accuracy_score(y_test_class, y_pred_ada)

print(f"\nAdaBoost Results:")
print(f"Test Accuracy: {ada_accuracy:.4f}")
print(f"Number of weak learners used: {len(ada_boost.estimators)}")

# Plot learning curve
ada_boost.plot_learning_curve(X_train_class, y_train_class, X_test_class, y_test_class)

# Example 2: Gradient Boosting for Regression
print("\n" + "="*50)
print("2. GRADIENT BOOSTING REGRESSION EXAMPLE")
print("="*50)

# Generate regression data
print("\nGenerating regression data...")
X_reg, y_reg = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

print(f"Training data shape: {X_train_reg.shape}")
print(f"Test data shape: {X_test_reg.shape}")

# Train Gradient Boosting
print("\nTraining Gradient Boosting regressor...")
gb_regressor = SimpleGradientBoostingRegressor(
    n_estimators=80, 
    learning_rate=0.1, 
    max_depth=3
)
gb_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_gb = gb_regressor.predict(X_test_reg)
gb_mse = mean_squared_error(y_test_reg, y_pred_gb)

print(f"\nGradient Boosting Results:")
print(f"Test MSE: {gb_mse:.4f}")
print(f"Number of estimators: {len(gb_regressor.estimators)}")

# Plot learning curve
gb_regressor.plot_learning_curve(X_train_reg, y_train_reg, X_test_reg, y_test_reg)

# Comparison with sklearn implementations
print("\n" + "="*50)
print("3. COMPARISON WITH SKLEARN")
print("="*50)

# Compare AdaBoost
print("\nComparing AdaBoost with sklearn...")
sklearn_ada = AdaBoostClassifier(n_estimators=30, random_state=42)
sklearn_ada.fit(X_train_class, y_train_class)
sklearn_ada_pred = sklearn_ada.predict(X_test_class)
sklearn_ada_accuracy = accuracy_score(y_test_class, sklearn_ada_pred)

print(f"Our AdaBoost accuracy: {ada_accuracy:.4f}")
print(f"Sklearn AdaBoost accuracy: {sklearn_ada_accuracy:.4f}")

# Compare Gradient Boosting
print("\nComparing Gradient Boosting with sklearn...")
sklearn_gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, 
                                      max_depth=3, random_state=42)
sklearn_gb.fit(X_train_reg, y_train_reg)
sklearn_gb_pred = sklearn_gb.predict(X_test_reg)
sklearn_gb_mse = mean_squared_error(y_test_reg, sklearn_gb_pred)

print(f"Our Gradient Boosting MSE: {gb_mse:.4f}")
print(f"Sklearn Gradient Boosting MSE: {sklearn_gb_mse:.4f}")

print("\n" + "="*70)
print("BOOSTING ALGORITHMS SUMMARY")
print("="*70)
print("\nADABOOST (Classification):")
print("1. Start with equal sample weights")
print("2. Train weak learner on weighted samples")
print("3. Calculate error and learner weight")
print("4. Increase weights for misclassified samples")
print("5. Final prediction = weighted vote")
print("\nGRADIENT BOOSTING (Regression):")
print("1. Start with mean prediction")
print("2. Calculate residuals (what we got wrong)")
print("3. Train weak learner to predict residuals")
print("4. Add scaled prediction to current model")
print("5. Final prediction = sum of all contributions")
print("\nKey Insight: Both methods learn from mistakes!")
print("="*70)