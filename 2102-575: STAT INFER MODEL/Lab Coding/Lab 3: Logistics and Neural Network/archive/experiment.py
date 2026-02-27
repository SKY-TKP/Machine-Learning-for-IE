import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
import os

# Data processing functions from the notebook
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    m            = len(df)
    perm         = np.random.permutation(m)
    train_end    = int(np.floor(int(train_percent * m)))
    validate_end = int(np.floor(int(validate_percent * m) + train_end))

    train        = df.iloc[perm[:train_end]].copy()
    validate     = df.iloc[perm[train_end:validate_end]].copy()
    test         = df.iloc[perm[validate_end:]].copy()

    train       = train.reset_index(drop=True)
    validate    = validate.reset_index(drop=True)
    test        = test.reset_index(drop=True)

    return train, validate, test

class CustomDataset(Dataset):
    def __init__(self, df, target_column, feat_column):
        self.x = torch.tensor(df[feat_column].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.ep      = 0
        self.min_validation_loss = float('inf')
        self.weight  = None

    def check(self, weight, ep, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.weight = weight
            self.ep     = ep
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs=1, n_hidden=16, num_layers=1, learning_rate=0.001, max_epoch=100, model_folder="models", patience=5):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(n_inputs, n_outputs))
        elif num_layers == 2:
            layers.append(nn.Linear(n_inputs, n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, n_outputs))
        else: # num_layers >= 3
            layers.append(nn.Linear(n_inputs, n_hidden))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, n_outputs))
        
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        
        self.max_epoch = max_epoch
        self.criterion = nn.BCELoss()
        self.F1_score = BinaryF1Score()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_folder = model_folder
        os.makedirs(self.model_folder, exist_ok=True)
        self.Earlystoper = EarlyStopper(patience=patience)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_set, valid_set, threshold, batch_size):
        train_bce_history, train_f1_history = [], []
        valid_bce_history, valid_f1_history = [], []
        
        validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)
        
        for epoch in range(self.max_epoch):
            trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
            self.model.train()
            
            epoch_train_loss, epoch_train_f1 = [], []
            for inputs, labels in trainloader:
                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss.append(loss.item())
                pred_binary = (predictions > threshold).float()
                f1 = self.F1_score(pred_binary, labels)
                epoch_train_f1.append(f1.item())
            
            train_bce_history.append(np.mean(epoch_train_loss))
            train_f1_history.append(np.mean(epoch_train_f1))
            
            self.model.eval()
            epoch_valid_loss, epoch_valid_f1 = [], []
            with torch.no_grad():
                for inputs, labels in validloader:
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, labels)
                    epoch_valid_loss.append(loss.item())
                    pred_binary = (predictions >= threshold).float()
                    f1 = self.F1_score(pred_binary, labels)
                    epoch_valid_f1.append(f1.item())
            
            avg_valid_loss = np.mean(epoch_valid_loss)
            valid_bce_history.append(avg_valid_loss)
            valid_f1_history.append(np.mean(epoch_valid_f1))
            
            if self.Earlystoper.check(self.model.state_dict(), epoch, avg_valid_loss):
                break
                
        return train_bce_history, train_f1_history, valid_bce_history, valid_f1_history

# Load and preprocess data
filepath = "https://raw.githubusercontent.com/SKY-TKP/Machine-Learning-for-IE/refs/heads/main/2102-575%3A%20STAT%20INFER%20MODEL/Lab%20Coding/Lab%203%3A%20Logistics%20and%20Neural%20Network/archive/Student%20Depression%20Dataset.csv"
df = pd.read_csv(filepath)
target_column = ['Depression']
feat_column = ['Gender', 'Age', 'City', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']

df = df[target_column + feat_column].dropna()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

train_df, test_df, valid_df = train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=42)

scaler = StandardScaler()
train_df[feat_column] = scaler.fit_transform(train_df[feat_column])
valid_df[feat_column] = scaler.transform(valid_df[feat_column])
test_df[feat_column] = scaler.transform(test_df[feat_column])

train_set = CustomDataset(train_df, target_column, feat_column)
valid_set = CustomDataset(valid_df, target_column, feat_column)

# Run experiments
results = {}
for layers in [1, 2, 3]:
    print(f"Training model with {layers} layers...")
    model = NeuralNetwork(n_inputs=len(feat_column), num_layers=layers, n_hidden=16, learning_rate=0.001, max_epoch=50, patience=5)
    hist = model.train_model(train_set, valid_set, threshold=0.5, batch_size=256)
    results[layers] = hist

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# BCE Loss Plot
for layers, hist in results.items():
    axes[0].plot(hist[0], label=f'Train {layers} layers')
    axes[0].plot(hist[2], '--', label=f'Valid {layers} layers')
axes[0].set_title('BCE Loss vs Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('BCE Loss')
axes[0].legend()

# F1 Score Plot
for layers, hist in results.items():
    axes[1].plot(hist[1], label=f'Train {layers} layers')
    axes[1].plot(hist[3], '--', label=f'Valid {layers} layers')
axes[1].set_title('F1 Score vs Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('F1 Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('results_plot.png')
print("Plots saved to results_plot.png")

# Print final metrics for comparison
for layers, hist in results.items():
    print(f"Layers: {layers} | Final Valid BCE: {hist[2][-1]:.4f} | Final Valid F1: {hist[3][-1]:.4f}")
