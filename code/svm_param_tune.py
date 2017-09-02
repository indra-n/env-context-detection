import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn import svm
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler

cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

# cols = ['num_sat_25', 'sum_snr_25',
#         'num_sat', 'sum_snr',
#         'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
#         'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)


train_data = train_df
test_data = test_df

print(train_data['true_class'].value_counts())

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])


param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001]
}

cv_generator = GroupKFold(n_splits=3)
for i in range(50):
    CV_rfc = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv_generator, n_jobs=1, iid=False) #, scoring='f1_weighted')
    CV_rfc.fit(train_input, train_data['true_class'], groups=train_data['location_id'])
    save_string = "..\saves\cv3_group_svm_rad_model_aws" + str(i) + ".pickle"
    # CV_result_list.append(CV_rfc)
    with open(save_string, 'wb') as f:
        pickle.dump(CV_rfc, f)
    print(CV_rfc.best_params_)

print("finished")






