import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
import csv

# Load the data

cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)

# Experiment with Han's data
with open('..\saves\given_data_sample_full.pickle', 'rb') as f:
    given_data = pickle.load(f)

inter_train_df = [train_df, test_df]
inter_train = pd.concat(inter_train_df).sample(frac=0.5).reset_index(drop=True)

train_data = inter_train
test_data = given_data

# train_data = train_df
# test_data = test_df

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])

C_def = 1
gamma_def = 1 # 1/num_features
clf = svm.SVC(C=10, gamma=10)
clf.fit(train_input, train_data['true_class'])

test_string = "file_" + str(C_def) + ".f"
print(test_string)

pre = clf.predict(test_input)

differ = abs(pre - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print(accu)

cm = confusion_matrix(test_data['true_class'], pre)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)



