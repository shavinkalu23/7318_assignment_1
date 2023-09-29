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
from sklearn.metrics import roc_auc_score

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
    def __init__(self, input_size, hidden_sizes, output_size,activation):
        super(MultiLayerPerceptron, self).__init__()
        self.activation = activation
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())  # You can use other activation functions as well
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        out = torch.sigmoid(self.model(x))
        out = self.activation(out)
        return out




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

    
dataset = MyDataset('/Users/a1904121/Library/CloudStorage/GoogleDrive-a1904121@adelaide.edu.au/Other computers/My MacBook Air/Adelaide Uni/2023 Trimester 3/Deep Learning Fundamentals/Assignment 1/diabetes_scale.txt')


# Split dataset into training, validation, and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


# Set fixed random number seed to reproduce results
torch.manual_seed(42) 

# Define hyperparameters to tune
learning_rates = [ 0.001, 0.01, 0.1, 1]
epochs =50 #, 100, 200]
optimizers = ['SGD', 'Adam', 'RMSprop']
# Define different activation functions
activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
batch_sizes = [10, 30, 90]

slp_best_accuracy = 0
slp_best_lr = 0
slp_best_epochs = 0

mlp_best_accuracy = 0
mlp_best_lr = 0
mlp_best_epochs = 0

# Perform k-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)

# MLP
hidden_sizes = [16]  
output_size = 1

# Create an empty list to store the results
results_list= []

#Coarse tuning (small epochs)
for activation in activations:
    for optimizer_name in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:
                slp_avg_accuracy = 0
                mlp_avg_accuracy = 0
        
                for train_indices, val_indices in kf.split(train_dataset):
                    train_val_dataset = torch.utils.data.Subset(train_dataset, train_indices)
                    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        
                    train_val_loader = DataLoader(train_val_dataset, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        
                    # Define model, loss function, and optimizer
                    slp_model = Perceptron(len(dataset[0][0]),activation)
                    
                    slp_optimizer = getattr(torch.optim, optimizer_name)(slp_model.parameters(), lr=lr)
                    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
                    #Use same loss_fn for SLP and MLP 
                    loss_fn= nn.BCELoss()
                    # Define model, loss function, and optimizer
                    mlp_model = MultiLayerPerceptron(len(dataset[0][0]),hidden_sizes, output_size, activation)
                    mlp_optimizer = getattr(torch.optim, optimizer_name)(mlp_model.parameters(), lr=lr)
        
        
        
                    # Train the model on the training set
                    for epoch in range(epochs):
                        train_loop(train_val_loader, slp_model, loss_fn, slp_optimizer)
                        train_loop(train_val_loader, mlp_model, loss_fn, mlp_optimizer)
                    # Evaluate on the validation set
                    slp_accuracy = test_loop(val_loader, slp_model, loss_fn)
                    mlp_accuracy = test_loop(val_loader, mlp_model, loss_fn)
                    
                    
                    slp_avg_accuracy += slp_accuracy
                    mlp_avg_accuracy += mlp_accuracy
                    
                    

        
                slp_avg_accuracy /= num_folds
                mlp_avg_accuracy /= num_folds
                
                # Append results to the list
                results_list.append({
                'Activation': activation,
                'Optimizer': optimizer_name,
                'Learning Rate': lr,
                'Batch Size': bs,
                'SLP Accuracy': slp_avg_accuracy,
                'MLP Accuracy': mlp_avg_accuracy})
                # Check if this combination of hyperparameters gives better average accuracy
                if slp_avg_accuracy > slp_best_accuracy:
                    slp_best_accuracy = slp_avg_accuracy
                    best_lr = lr
                    #best_epochs = epochs
                    slp_best_activation = activation
                    slp_best_optimizer = optimizer_name
                    slp_best_batch_size = bs
                    
                if mlp_avg_accuracy > mlp_best_accuracy:
                    mlp_best_accuracy = mlp_avg_accuracy
                    mlp_best_lr = lr
                    #best_epochs = epochs
                    mlp_best_activation = activation
                    mlp_best_optimizer = optimizer_name
                    mlp_best_batch_size = bs

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results_list)                     

model = slp_model
best_batch_size = slp_best_batch_size
best_lr = slp_best_lr
best_activation = slp_best_activation
best_optimizer = slp_best_optimizer

#uncomment to use MLP instead of SLP for fine tuning
# model = mlp_model
# best_batch_size = mlp_best_batch_size
# best_lr = mlp_best_lr
# best_activation = mlp_best_activation
# best_optimizer = mlp_best_optimizer

                
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
best_accuracy = 0
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

# Train best model on entire training dataset
for epoch in range(epochs):
    for features, labels in train_loader:
        train_loop(train_loader, final_model, loss_fn, final_optimizer)

# Evaluate the final model on the test set
test_accuracy = test_loop(test_loader,final_model,loss_fn)

print(f'Best learning rate: {best_lr}')
print(f'Best batch size: {best_batch_size}')
print(f'Best activation function: {best_activation}')
print(f'Best optimizer: {best_optimizer}')
print(f'Training Accuracy: {best_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')



# Assuming final_model is your trained Perceptron model and test_loader is your DataLoader for the test set

final_model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = final_model(features)
        y_true.extend(labels.numpy())
        y_scores.extend(outputs.numpy())

y_true = torch.tensor(y_true)
y_scores = torch.tensor(y_scores)

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)
print(f'AUC: {auc:.4f}')







