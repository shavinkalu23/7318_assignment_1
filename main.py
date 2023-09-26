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
import matplotlib.pyplot as plt
# Define the Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size, activation):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = activation
    
    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        out = self.activation(out)
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
    for features, labels in dataloader:
        outputs = model(features)
        loss = loss_fn(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

            
            
def test_loop(dataloader, model, loss_fn):
        model.eval()
        with torch.no_grad(): #Ensure no gradients are computed during testing
            correct = 0
            total = 0
            for features, labels in dataloader:
                outputs = model(features)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels.unsqueeze(1)).sum().item()

            accuracy = correct / total
            return accuracy

    
dataset = MyDataset('/Users/shavinkalu/Adelaide Uni/2023 Trimester 3/Deep Learning Fundamentals/Assignment 1/diabetes_scale.txt')


# Split dataset into training, validation, and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


# Set fixed random number seed to reproduce results
torch.manual_seed(42) 

# Define hyperparameters to tune
learning_rates = [0.001, 0.01, 0.1, 1]
num_epochs = [50] #, 100, 200]
optimizers = ['SGD', 'Adam', 'RMSprop']
# Define different activation functions
activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
batch_sizes = [10, 30, 90]

best_accuracy = 0
best_lr = 0
best_epochs = 0

# Perform k-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)


#Coarse tuning (small epochs)
for activation in activations:
    for optimizer_name in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:
                for epochs in num_epochs:
                    avg_accuracy = 0
            
                    for train_indices, val_indices in kf.split(train_dataset):
                        train_val_dataset = torch.utils.data.Subset(train_dataset, train_indices)
                        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            
                        train_val_loader = DataLoader(train_val_dataset, batch_size=bs, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            
                        # Define model, loss function, and optimizer
                        model = Perceptron(len(dataset[0][0]),activation)
                        loss_fn= nn.BCELoss()
                        
                        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
                        #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            
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
                        #best_epochs = epochs
                        best_activation = activation
                        best_optimizer = optimizer_name
                        best_batch_size = bs
 
#Fine tuning
epochs = 200 #, 100, 200]
# Define hyperparameters to tune
learning_rates = [0.05, 0.1, 0.5]   
losses = np.array([])
losses_array = np.zeros((epochs,len(learning_rates)))
train_accuracies_array =  np.zeros((epochs,len(learning_rates)))
val_accuracies_array =  np.zeros((epochs,len(learning_rates)))

i=0
# Lists to store training and validation accuracies
for lr in learning_rates:
    
    avg_accuracy = 0
    losses = np.array([])
    
    train_accuracies = np.array([])
    val_accuracies = np.array([])
    
    for train_indices, val_indices in kf.split(train_dataset):
        
        train_val_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

        train_val_loader = DataLoader(train_val_dataset, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

        # Define model, loss function, and optimizer
        model = Perceptron(len(dataset[0][0]),best_activation)
        loss_fn= nn.BCELoss()
        
        optimizer = getattr(torch.optim, best_optimizer)(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Train the model on the training set
        for epoch in range(epochs):
            loss= train_loop(train_val_loader, model, loss_fn, optimizer)
            losses = np.append(losses, loss)
            
            train_accuracy = test_loop(train_val_loader, model, loss_fn)
            # Evaluate on the validation set
            val_accuracy = test_loop(val_loader, model, loss_fn)
            
            
            train_accuracies= np.append(train_accuracies,train_accuracy)
            val_accuracies= np.append(val_accuracies,val_accuracy)
            
        avg_accuracy += val_accuracy

    avg_accuracy /= num_folds
    
    # Check if this combination of hyperparameters gives better average accuracy
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_lr = lr

    
    # get mean accuracy over k folds for each epochs 
    losses_array[:,i] = np.mean(losses.reshape( -1, epochs), axis =0)
    train_accuracies_array[:,i] = np.mean(train_accuracies.reshape( -1, epochs), axis =0)
    val_accuracies_array[:,i] = np.mean(val_accuracies.reshape( -1, epochs), axis =0)
    i+=1



# Plotting losses
# Plotting
for i in range(len(learning_rates)):
    plt.plot(np.arange(epochs), losses_array[:,i], label=f'Loss - LR: {learning_rates[i]}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#train_accuracies = np.mean(losses_epoch.reshape(-1, epochs), axis =0)
# Plotting
i = learning_rates.index(best_lr)
plt.plot(np.arange(epochs), train_accuracies_array[:,i], label='Train Accuracy')
plt.plot(np.arange(epochs), val_accuracies_array[:,i], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
# Train the final model with the best hyperparameters on the combined training set
final_model = Perceptron(len(dataset[0][0]),best_activation)
final_optimizer = getattr(torch.optim, best_optimizer)(final_model.parameters(), lr=best_lr)

for epoch in range(best_epochs):
    for features, labels in train_loader:
        train_loop(train_loader, final_model, loss_fn, final_optimizer)

# Evaluate the final model on the test set
test_accuracy = test_loop(test_loader,model,loss_fn)

print(f'Best learning rate: {best_lr}')
print(f'Best batch size: {best_batch_size}')
print(f'Best activation function: {best_activation}')
print(f'Best optimizer: {best_optimizer}')
print(f'Training Accuracy: {best_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')