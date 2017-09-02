import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
import pickle

# Load the data
cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

# My test data
with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)

# # Han's test data
# with open('..\saves\given_data_test_set.pickle', 'rb') as f:
#     test_df = pickle.load(f)
# # Mixed train set for Han's data
# with open('..\saves\mixed_train_set.pickle', 'rb') as f:
#     train_df = pickle.load(f)

train_data = train_df
test_data = test_df

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])



forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3, random_state=57)
bayes = GaussianNB(priors=[0.25, 0.25, 0.25, 0.25])
svm = svm.SVC(probability=True, C=0.01, gamma=1, random_state=1289)


clf = VotingClassifier(estimators=[('rf', forest), ('bayes', bayes), ('svm', svm)], voting='hard')
clf.fit(train_input, train_data['true_class'])

pred = clf.predict(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print("Voting accuracy:")
print(accu)

print("Voting confusion matrix")
cm = confusion_matrix(test_data['true_class'], pred)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)

# # Showing location ids for wrong predictions (my data only)
# print("Incorrect prediction counts")
# outcome = test_data.loc[:, ('true_class', 'location_id')]
# outcome['predicted_class'] = pred
# wrong_= outcome[differ != 0]
# print(wrong_.groupby(['location_id', 'predicted_class']).size())
