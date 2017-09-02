import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
import pickle
import csv

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    evening_data_train = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    morning_data_test = pickle.load(f)

full_data_set = pd.concat([evening_data_train, morning_data_test]).sample(frac=1).reset_index(drop=True)

indoor = full_data_set[full_data_set['true_class'] == 1]
inter = full_data_set[full_data_set['true_class'] == 2]
urban = full_data_set[full_data_set['true_class'] == 3]
open_sky = full_data_set[full_data_set['true_class'] == 4]

indoor_train = indoor.sample(1491)
indoor_test = indoor.drop(indoor_train.index)

inter_train = inter.sample(1382)
inter_test = inter.drop(inter_train.index)

urban_train = urban.sample(1215)
urban_test = urban.drop(urban_train.index)

open_train = open_sky.sample(1030)
open_test = open_sky.drop(open_train.index)

mixed_train_set = pd.concat([indoor_train, inter_train, urban_train, open_train]).sample(frac=1).reset_index(drop=True)
mixed_test_set = pd.concat([indoor_test, inter_test, urban_test, open_test]).sample(frac=1).reset_index(drop=True)

print(mixed_train_set.shape)
print(mixed_test_set.shape)

with open('..\saves\mixed_train_set.pickle', 'wb') as f:
    pickle.dump(mixed_train_set, f)

with open('..\saves\mixed_test_set.pickle', 'wb') as f:
    pickle.dump(mixed_test_set, f)