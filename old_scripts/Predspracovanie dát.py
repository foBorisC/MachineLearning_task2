# python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from Classificator import one_hot_encode, dataframe_to_floats, remove_outliers_iqr
from csv_stats import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw_data = read_csv('zadanie1-data.csv')
    df = pd.DataFrame(raw_data)

    print(df.shape)

    # Remove euribor3m column because 50% of values are missing
    # Remove previous column because 10k of 0.0 values
    df = df.drop('euribor3m', axis=1)
    df = df.drop('previous', axis=1)
    df = df.drop('duration', axis=1)
    df = df.drop('pdays', axis=1)
    df = df.drop('day_of_week', axis=1)

    # One hot encode for categorical columns
    df = one_hot_encode(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',  'poutcome', 'subscribed'])
    # all object columns to float
    df = dataframe_to_floats(df)
    # remove rows with missing values
    df_clean = df.dropna()
    print("Dáta pred odstránením NA:", df.shape)
    print("Dáta po odstránení NA:", df_clean.shape)

    # remove outliers
    df_clean = remove_outliers_iqr(df_clean, [ 'age', 'campaign', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate','nr.employed'])
    print(df_clean)