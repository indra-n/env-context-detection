import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the data

cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

### Select training and test sets
# My training data
with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)
# Load level 1 training data
with open('..\saves\meta_train_data_2.pickle', 'rb') as f:
    meta_train_data = pickle.load(f)

with open('..\saves\meta_train_labels_2.pickle', 'rb') as f:
    meta_train_true_labels = pickle.load(f)

# ### Mixed train set for testing with Han's data
# with open('..\saves\mixed_train_set.pickle', 'rb') as f:
#     train_df = pickle.load(f)
# #### Meta level training data coresponding to the mixed train set
# with open('..\saves\meta_train_data_mixed.pickle', 'rb') as f:
#     meta_train_data = pickle.load(f)
#
# with open('..\saves\meta_train_labels_mixed.pickle', 'rb') as f:
#     meta_train_true_labels = pickle.load(f)


# My test data
with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)


#### Han's test data
# with open('..\saves\given_data_test_set.pickle', 'rb') as f:
#     test_df = pickle.load(f)

#######

### Stacking meta classifier
#last_clf = LogisticRegression(C=0.01, solver='liblinear')
# last_clf = KNeighborsClassifier(n_neighbors=200)
last_clf = RadiusNeighborsClassifier(radius=1.7, outlier_label=2)

#####

train_data = train_df
test_data = test_df


scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])

####################################################
# cols_iid = ['elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
#         'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25',
#         'elev_0_30_weak', 'elev_30_60_weak', 'elev_60_90_weak',
#         'sum_elev_0_30_weak', 'sum_elev_30_60_weak', 'sum_elev_60_90_weak']
# scaler_iid = MinMaxScaler()
# scaler_iid.partial_fit(train_data[cols_iid])
# scaler_iid.partial_fit(test_data[cols_iid])
# train_input_iid = scaler_iid.transform(train_data[cols_iid])
# test_input_iid = scaler_iid.transform(test_data[cols_iid])

forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3, random_state=57)
bayes = GaussianNB(priors=[0.25, 0.25, 0.25, 0.25])
svm = svm.SVC(probability=True, C=0.01, gamma=1, random_state=1289)

last_clf.fit(meta_train_data, meta_train_true_labels)

forest.fit(train_input, train_data['true_class'])
bayes.fit(train_input, train_data['true_class'])
#bayes.fit(train_input_iid, train_data['true_class'])
svm.fit(train_input, train_data['true_class'])


test_proba_forest = forest.predict_proba(test_input)
test_proba_bayes = bayes.predict_proba(test_input)
#test_proba_bayes = bayes.predict_proba(test_input_iid)
test_proba_svm = svm.predict_proba(test_input)

test_proba = np.concatenate((test_proba_forest, test_proba_bayes, test_proba_svm), axis=1)
# test_proba = np.concatenate((test_proba_forest, test_proba_svm), axis=1)
# test_proba = (test_proba_forest + test_proba_svm + test_proba_bayes)/3


pred = last_clf.predict(test_proba)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print("Stacked accuracy:")
print(accu)

print("Stacked confusion matrix")
cm = confusion_matrix(test_data['true_class'], pred)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)

# Showing location ids for wrong predictions (my data only)
print("Incorrect prediction counts for Stacked")
outcome = test_data.loc[:, ('true_class', 'location_id')]
outcome['predicted_class'] = pred
wrong_= outcome[differ != 0]
print(wrong_.groupby(['location_id', 'predicted_class']).size())

#### Forest ####
pred = forest.predict(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print("Forest accuracy:")
print(accu)

cm = confusion_matrix(test_data['true_class'], pred)
print("Forest confusion matrix")
print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
print(cm_proc)

# Showing location ids for wrong predictions (my data only)
print("Incorrect prediction counts for Forest")
outcome = test_data.loc[:, ('true_class', 'location_id')]
outcome['predicted_class'] = pred
wrong_= outcome[differ != 0]
print(wrong_.groupby(['location_id', 'predicted_class']).size())

#### SVM ####
pred = svm.predict(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print("SVM accuracy:")
print(accu)
cm = confusion_matrix(test_data['true_class'], pred)
print("SVM confusion matrix")
print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
print(cm_proc)

# Showing location ids for wrong predictions (my data only)
print("Incorrect prediction counts for SVM")
outcome = test_data.loc[:, ('true_class', 'location_id')]
outcome['predicted_class'] = pred
wrong_= outcome[differ != 0]
print(wrong_.groupby(['location_id', 'predicted_class']).size())


#### Bayes ######
pred = bayes.predict(test_input)
#pred = bayes.predict(test_input_iid)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print("Bayes accuracy:")
print(accu)

cm = confusion_matrix(test_data['true_class'], pred)
print("Bayes confusion matrix")
print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
#print(cm_proc)

# Showing location ids for wrong predictions (my data only)
print("Incorrect prediction counts for Bayes")
outcome = test_data.loc[:, ('true_class', 'location_id')]
outcome['predicted_class'] = pred
wrong_= outcome[differ != 0]
print(wrong_.groupby(['location_id', 'predicted_class']).size())