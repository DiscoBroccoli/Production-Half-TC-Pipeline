# import evaluation
from dataset.Production_prep import ProductionDataset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow import keras
import seaborn as sns

P = ProductionDataset()

x_train_standard = P._set_standardization_values(test=False)
x_test_standard = P._set_standardization_values(test=True)

y_train = P.y_train
y_test = P.y_test

train_stats = x_train_standard.describe()
train_stats = train_stats.transpose()
train_stats

"""
sns.pairplot(x_train_standard[["uu", "uv", "vv", "ww"]], diag_kind="kde")
sns.pairplot(x_train_standard[["rhou", "rhov", "rho", "yplus"]], diag_kind="kde")
"""





