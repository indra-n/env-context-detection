import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle

# selected
cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

# independent
cols_iid = ['elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25',
        'elev_0_30_weak', 'elev_30_60_weak', 'elev_60_90_weak',
         'sum_elev_0_30_weak', 'sum_elev_30_60_weak', 'sum_elev_60_90_weak']

# all
# cols = ['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'num_sat_u20', 'sum_sat_u20', 'elev_0_30', 'elev_30_60',
# 'elev_60_90', 'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90', 'elev_0_30_25', 'elev_30_60_25',
# 'elev_60_90_25', 'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25', 'elev_0_30_weak', 'elev_30_60_weak',
#  'elev_60_90_weak', 'sum_elev_0_30_weak', 'sum_elev_30_60_weak', 'sum_elev_60_90_weak']


with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)

train_data = train_df
test_data = test_df

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])

scaler_iid = MinMaxScaler()
scaler_iid.partial_fit(train_data[cols_iid])
scaler_iid.partial_fit(test_data[cols_iid])
train_input_iid = scaler_iid.transform(train_data[cols_iid])
test_input_iid = scaler_iid.transform(test_data[cols_iid])

cv_generator = LeaveOneGroupOut()
#cv_generator = StratifiedKFold(n_splits=10)
cv_score = cross_val_score(GaussianNB(priors=[0.25, 0.25, 0.25, 0.25]), train_input, train_data['true_class'], groups=train_data['location_id'], cv=cv_generator)
print("Bayes LOGO CV score for full set")
print(cv_score)
print(cv_score.mean())

cv_score_iid = cross_val_score(GaussianNB(priors=[0.25, 0.25, 0.25, 0.25]), train_input_iid, train_data['true_class'], groups=train_data['location_id'], cv=cv_generator)
print("Bayes LOGO CV score for reduced set")
print(cv_score_iid)
print(cv_score_iid.mean())

clf = GaussianNB()
clf.fit(train_input, train_data['true_class'])

pred = clf.predict(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print(accu)

cm = confusion_matrix(test_data['true_class'], pred)
print(cm)

cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)

# with open('..\saves\\bayes_model1.pickle', 'wb') as f:
#     pickle.dump(clf, f)

