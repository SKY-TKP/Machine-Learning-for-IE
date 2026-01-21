# ==========================================
# Homework 1: Clustering and Regression
# Part: Titanic Logistic Regression (T8 - T13)
# Name: Thananop Kullapan
# Student ID: 6530182121
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

try:
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# ==========================================
# 2. Data Preprocessing (T8, T9)
# ==========================================

# T8: Fill missing Age with Median
median_age = train["Age"].median()
train["Age"] = train["Age"].fillna(median_age)
test["Age"] = test["Age"].fillna(median_age)
print(f"[T8] Median Age used for filling: {median_age}")

# T9: Handle Categorical Data
# Fill missing Embarked with Mode
mode_embarked = train["Embarked"].mode()[0]
train["Embarked"] = train["Embarked"].fillna(mode_embarked)
test["Embarked"] = test["Embarked"].fillna(mode_embarked)
print(f"[T9] Mode Embarked used for filling: {mode_embarked}")

def preprocess_features(df):
    """
    Map categorical features to numbers as per assignment instructions.
    Sex: male=0, female=1
    Embarked: S=0, C=1, Q=2
    """
    df = df.copy()
    # Mapping Sex
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1
    
    # Mapping Embarked
    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "C", "Embarked"] = 1
    df.loc[df["Embarked"] == "Q", "Embarked"] = 2
    
    return df

train_proc = preprocess_features(train)
test_proc = preprocess_features(test)

# Select features
features = ["Pclass", "Sex", "Age", "Embarked"]

# Convert to Numpy Arrays
X_train = np.array(train_proc[features].values, dtype=float)
y_train = np.array(train_proc["Survived"].values, dtype=float)
X_test = np.array(test_proc[features].values, dtype=float)

# *** IMPORTANT: Normalization ***
# Gradient Descent performs poorly without feature scaling
mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)

# Normalize both train and test using train statistics
X_train_norm = (X_train - mean_X) / (std_X + 1e-8)
X_test_norm = (X_test - mean_X) / (std_X + 1e-8)

# ==========================================
# 3. Logistic Regression Implementation (T10)
# ==========================================

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=5000):
        self.lr = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.cost_history = []
        
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.iterations):
            # Linear Prediction
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear_model)
            
            # Gradient Calculation
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update Parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Optional: Cost Logging (Cross Entropy)
            # cost = -np.mean(y * np.log(y_pred+1e-9) + (1-y) * np.log(1-y_pred+1e-9))
            # self.cost_history.append(cost)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        y_probs = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in y_probs])

# Train Model
print("\nTraining Logistic Regression...")
model = LogisticRegressionGD(learning_rate=0.01, iterations=10000)
model.fit(X_train_norm, y_train)

# Evaluate on Training Set
train_preds = model.predict(X_train_norm)
train_acc = np.mean(train_preds == y_train)
print(f"[T10] Training Accuracy: {train_acc:.4f}")
print(f"Learned Weights: {model.w}")
print(f"Learned Bias: {model.b}")

# ==========================================
# 4. Submission File Generation (T11)
# ==========================================

test_preds = model.predict(X_test_norm)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})

submission_filename = "titanic_submission_6530182121.csv"
submission.to_csv(submission_filename, index=False)
print(f"\n[T11] Submission file created: {submission_filename}")
print("Please upload this file to Kaggle and take a screenshot of the score.")

# ==========================================
# 5. Higher Order Features (T12)
# ==========================================
print("\n--- T12: Higher Order Features Experiment (Adding Age^2) ---")

# Adding Age^2 as a new feature
X_train_poly = np.c_[X_train, X_train[:, 2]**2] 
# Re-normalize
mean_poly = np.mean(X_train_poly, axis=0)
std_poly = np.std(X_train_poly, axis=0)
X_train_poly_norm = (X_train_poly - mean_poly) / (std_poly + 1e-8)

model_poly = LogisticRegressionGD(learning_rate=0.01, iterations=10000)
model_poly.fit(X_train_poly_norm, y_train)
poly_acc = np.mean(model_poly.predict(X_train_poly_norm) == y_train)

print(f"Accuracy with Higher Order Features: {poly_acc:.4f}")
if poly_acc > train_acc:
    print("Result: Accuracy improved.")
else:
    print("Result: Accuracy did not improve significantly (or overfitting occurred).")
