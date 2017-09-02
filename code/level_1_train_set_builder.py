import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Creates training data for the meta layer classifier


locations = [214, 324, 514, 534, 614, 624, 724, 744, 754, 814, 844, 1014, 1024, 1114, 1124, 1134, 1214, 1314, 1414]

cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

# Need to also load test data for scaling purposes
# with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
#     train_df = pickle.load(f)
#
# with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
#     test_df = pickle.load(f)

with open('..\saves\mixed_train_set.pickle', 'rb') as f:
    train_df = pickle.load(f)
with open('..\saves\mixed_test_set.pickle', 'rb') as f:
    test_df = pickle.load(f)

train_data = train_df
test_data = test_df

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])

# Models with best parameters
forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3)
bayes = GaussianNB(priors=[0.25, 0.25, 0.25, 0.25])
svm = svm.SVC(probability=True, C=0.01, gamma=1)

train_pred_X = []
train_true_labels = []
train_groups = []
train_true_class = train_data['true_class'].values
train_group_id = train_data['location_id'].values

logo = LeaveOneGroupOut()
# logo = StratifiedKFold(n_splits=10)

fold = 1
for train_index, val_index in logo.split(train_input, train_true_class, train_group_id):
    X_train, X_test = train_input[train_index], train_input[val_index]
    y_train, y_test = train_true_class[train_index], train_true_class[val_index]
    # fit base classifiers on training data
    # and get predictions for test data
    forest.fit(X_train, y_train)
    pred_f = forest.predict_proba(X_test)
    bayes.fit(X_train, y_train)
    pred_b = bayes.predict_proba(X_test)
    svm.fit(X_train, y_train)
    pred_s = svm.predict_proba(X_test)
    #
    pred_features = np.concatenate((pred_f, pred_b, pred_s), axis=1)
    # pred_features = np.concatenate((pred_f, pred_s), axis=1)
    # pred_features = (pred_f + pred_s + pred_b)/3
    train_pred_X.append(pred_features)
    train_true_labels.append(y_test)
    train_groups.append(train_group_id[val_index])
    print("Fold " + str(fold) + "done")
    fold = fold + 1

train_level_one_data = np.concatenate(train_pred_X, axis=0)
train_level_one_labels = np.concatenate(train_true_labels)
train_level_one_groups = np.concatenate(train_groups)

with open('..\saves\meta_train_data_mixed.pickle', 'wb') as f:
    pickle.dump(train_level_one_data, f)

with open('..\saves\meta_train_labels_mixed.pickle', 'wb') as f:
    pickle.dump(train_level_one_labels, f)

with open('..\saves\meta_train_groups_mixed.pickle', 'wb') as f:
    pickle.dump(train_level_one_groups, f)
