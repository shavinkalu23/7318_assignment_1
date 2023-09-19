#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:04:19 2023

@author: shavinkalu
"""

import pandas as pd
import re
import torch
import numpy as np
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
#Read normialised data file


# Open and read the file
with open('/Users/shavinkalu/Adelaide Uni/2023 Trimester 3/Deep Learning Fundamentals/Assignment 1/diabetes_scale.txt', 'r') as file:
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
df = pd.DataFrame(parsed_data, columns=['Label', 'Features'])

# Expand the Features column into separate columns
df = pd.concat([df.drop(['Features'], axis=1), df['Features'].apply(pd.Series).astype(float)], axis=1)

# Convert features to tensors
X = torch.tensor(df[df.columns[1:]].values, dtype = torch.float32) 

# Convert labels to tensor
y = torch.tensor(df[df.columns[0]].values, dtype = torch.float32)


class MyDataset(Dataset):
    def __init__(self, path):
        self.data = 
        
    def __getitem__(self, index)