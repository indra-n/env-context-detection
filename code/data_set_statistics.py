import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
import pickle
import csv

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    evening_data_train = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    morning_data_test = pickle.load(f)

# with open('..\saves\given_data_sample_full.pickle', 'rb') as f:
#     Hans_data = pickle.load(f)
with open('..\saves\given_data_test_set.pickle', 'rb') as f:
    Hans_data = pickle.load(f)

with open('..\saves\mixed_train_set.pickle', 'rb') as f:
    mixed_train_set = pickle.load(f)

train_groups = evening_data_train['true_class'].value_counts()
print("Training data split")
print(train_groups)
print("Training data size")
print(evening_data_train.shape)
print(type(train_groups))
train_groups_ration = train_groups/evening_data_train.shape[0]
print("Training data ratios")
print(train_groups_ration)

print()

test_groups = morning_data_test['true_class'].value_counts()
print("Test data split")
print(test_groups)
print("Test data size")
print(morning_data_test.shape)
print(type(test_groups))
test_groups_ration = test_groups/morning_data_test.shape[0]
print("Test data ratios")
print(test_groups_ration)

print()

hans_groups = Hans_data['true_class'].value_counts()
print("Han's data split")
print(hans_groups)
print("Han's data size")
print(Hans_data.shape)
print(type(hans_groups))
hans_groups_ration = hans_groups/Hans_data.shape[0]
print("Han's data ratios")
print(hans_groups_ration)

print()

mixed_train_groups = mixed_train_set['true_class'].value_counts()
print("Mixed training data split")
print(mixed_train_groups)
print("Mixed training data size")
print(mixed_train_set.shape)
print(type(mixed_train_groups))
mixed_train_groups_ration = mixed_train_groups/mixed_train_set.shape[0]
print("Mixed training data ratios")
print(mixed_train_groups_ration)
