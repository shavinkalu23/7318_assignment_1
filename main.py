#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:04:19 2023

@author: shavinkalu
"""

import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

# Define the Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerPerceptron, self).__init__()
        
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())  # You can use other activation functions as well
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Usage example:
input_size = 8  # Assuming 8 input features
hidden_sizes = [16, 8]  # Example: Two hidden layers with 16 and 8 units
output_size = 1

model = MultiLayerPerceptron(input_size, hidden_sizes, output_size)


#Read normialised data file
class MyDataset(Dataset):
    def __init__(self, path):
        # Open and read the file
        with open(path, 'r') as file:
            lines = file.readlines()

        # parse the data to a pandas dataframe
        def parse_data(line):
            label, features = line.split(' ', 1)
            label = int(label)
            features = dict(re.findall(r'(\d+):([\d.-]+)', features))
            return label, features

        # Parse the data and create a list of dictionaries
        parsed_data = [parse_data(line) for line in lines]

        # Create a DataFrame
        df = pd.DataFrame(parsed_data, columns=['label', 'features'])

        # Expand the Features column into separate columns
        df = pd.concat([df.drop(['features'], axis=1), df['features'].apply(pd.Series).astype(float)], axis=1)
        #convert labels to 0 and 1 instead of -1 to 1
        df['label'] = (df['label'].values>0).astype(float)
        df = df.dropna()
        # Convert features to tensors
        self.X = torch.tensor(df[df.columns[1:]].values, dtype=torch.float32) 

        # Convert labels to tensor
        self.y = torch.tensor(df['label'].values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
    

def train_loop(dataloader, model,loss_fn, optimizer):
    #size = len(dataloader.dataset)
    model.train()
    for features, labels in train_loader:
        outputs = model(features)
        loss = loss_fn(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

            
            
def test_loop(dataloader, model, loss_fn):
        model.eval()
        with torch.no_grad(): #Ensure no gradients are computed during testing
            correct = 0
            total = 0
            for features, labels in val_loader:
                outputs = model(features)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels.unsqueeze(1)).sum().item()

            accuracy = correct / total
            return accuracy

# set epoch of 50 to get rough idea of which parameters work

learning_rate = 0.5
batch_size = 10
epochs  = 50




    
dataset = MyDataset('/Users/shavinkalu/Adelaide Uni/2023 Trimester 3/Deep Learning Fundamentals/Assignment 1/diabetes_scale.txt')


# Split dataset into training, validation, and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
# 
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Set fixed random number seed
torch.manual_seed(42) 

# Define hyperparameters to tune
learning_rates = [0.001, 0.01, 0.1, 0.5, 1]
num_epochs = [50, 100, 200]


best_accuracy = 0
best_lr = 0
best_epochs = 0

# Perform k-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)



        

for lr in learning_rates:
    for epochs in num_epochs:
        avg_accuracy = 0

        for train_indices, val_indices in kf.split(train_dataset):
            train_val_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

            train_val_loader = DataLoader(train_val_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Define model, loss function, and optimizer
            model = Perceptron(len(dataset[0][0]))
            loss_fn= nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            # Train the model on the training set
            for epoch in range(epochs):
                train_loop(train_val_loader, model, loss_fn, optimizer)

            # Evaluate on the validation set
            accuracy = test_loop(val_loader, model, loss_fn)
                
            avg_accuracy += accuracy

        avg_accuracy /= num_folds

        # Check if this combination of hyperparameters gives better average accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_lr = lr
            best_epochs = epochs

# Train the final model with the best hyperparameters on the combined training set
final_model = Perceptron(len(dataset[0][0]))
final_optimizer = torch.optim.SGD(final_model.parameters(), lr=best_lr)

for epoch in range(best_epochs):
    for features, labels in train_loader:
        train_loop(train_val_loader, model, loss_fn, optimizer)

# Evaluate the final model on the test set
test_accuracy = test_loop(test_loader,model,loss_fn)

print(f'Best Learning Rate: {best_lr}')
print(f'Best Number of Epochs: {best_epochs}')
print(f'Training Accuracy: {best_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')