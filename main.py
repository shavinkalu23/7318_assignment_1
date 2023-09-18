#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:04:19 2023

@author: shavinkalu
"""

import pandas as pd
import re

# Define a function to parse the data
def parse_data(line):
    label, features = line.split(' ', 1)
    label = int(label)
    features = dict(re.findall(r'(\d+):([\d.]+)', features))
    return label, features

# Provided data
data = [
    "-1 1:6.000000 2:148.000000 3:72.000000 4:35.000000 5:0.000000 6:33.599998 7:0.627000 8:50.000000",
    "+1 1:1.000000 2:85.000000 3:66.000000 4:29.000000 5:0.000000 6:26.600000 7:0.351000 8:31.000000",
    "-1 1:8.000000 2:183.000000 3:64.000000 4:0.000000 5:0.000000 6:23.299999 7:0.672000 8:32.000000"
]

# Parse the data and create a list of dictionaries
parsed_data = [parse_data(line) for line in data]

# Create a DataFrame
df = pd.DataFrame(parsed_data, columns=['Label', 'Features'])

# Expand the Features column into separate columns
df = pd.concat([df.drop(['Features'], axis=1), df['Features'].apply(pd.Series).astype(float)], axis=1)

# Print the resulting DataFrame
print(df)
