import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
import pickle
import csv

# Load the data
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'elev_0_30', 'elev_30_60', 'elev_60_90',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']

locations = [214, 324, 514, 534, 614, 624, 724, 744, 754, 814, 844, 1014, 1024, 1114, 1124, 1134, 1214, 1314, 1414]

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
# cols = ['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25',
#         'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']

# with open('..\saves\\train_data1.pickle', 'rb') as f:
#     train_df = pickle.load(f)
#
# with open('..\saves\\test_data1.pickle', 'rb') as f:
#     test_df = pickle.load(f)

# with open('..\saves\\train_data_full_trimmed.pickle', 'rb') as f:
#     train_df = pickle.load(f)
#
# with open('..\saves\\test_data_full_trimmed.pickle', 'rb') as f:
#     test_df = pickle.load(f)
#
# train_data = train_df.loc[~train_df['location_id'].isin([834, 934, 724, 214, 424, 1134, 624, 114])]
# test_data = test_df.loc[~test_df['location_id'].isin([835, 935, 725, 215, 425, 1135, 625, 115])]

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)

# test_data = train_df.loc[train_df['location_id'].isin([624])]
# train_data = train_df.drop(test_data.index)
# train_data = train_df.sample(frac=0.5)
# test_data = train_df.drop(train_data.index)
train_data = train_df
test_data = test_df

# locations = train_data['location_id'].value_counts()
# print(locations.shape)

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])

# num_folds = 5

#
# for i in range(20):
#     group_kfold = GroupKFold(n_splits=num_folds)
#     j = 0
#     for train_index, test_index in group_kfold.split(train_input, train_data['true_class'], train_data['location_id']):
#         save_string = "..\cv_splits\cv_5fold_split" + str(i) + "_fold" + str(j) + "_train.pickle"
#         with open(save_string, 'wb') as f:
#             pickle.dump(train_index, f)
#         save_string = "..\cv_splits\cv_5fold_split" + str(i) + "_fold" + str(j) + "_test.pickle"
#         with open(save_string, 'wb') as f:
#             pickle.dump(test_index, f)
#         j = j + 1
# clf = svm.SVC(C=10, gamma=10)
# clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=3)
# mean_accu = 0
# for train_index, test_index in group_kfold.split(train_input, train_data['true_class'], train_data['location_id']):
#     X_train, X_test = train_input[train_index], train_input[test_index]
#     y_train, y_test = train_data['true_class'][train_index], train_data['true_class'][test_index]
#     clf.fit(X_train, y_train)
#     pre = clf.predict(X_test)
#     differ = abs(pre - y_test)
#     accu = 1 - np.count_nonzero(differ) / y_test.shape[0]
#     print(accu)
#     mean_accu = mean_accu + accu
#
#
# C_def = 1
# gamma_def = 1 # 1/num_features
#
#
#
# test_string = "file_" + str(C_def) + ".f"
# print(test_string)
# print(mean_accu/num_folds)
#
# pre = clf.predict(test_input)
#
# differ = abs(pre - test_data['true_class'])
# accu = 1 - np.count_nonzero(differ) / test_data.shape[0]
#
# print(accu)
#
# cm = confusion_matrix(test_data['true_class'], pre)
#
# print(cm)
# cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
# np.set_printoptions(suppress=True)
# print(cm_proc)

# with open('..\saves\\svm_model1.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# Forest
#clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=1)

n_estimators = [100, 200, 300, 400, 500]
max_features = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

scores = np.zeros((50, 1))
estimators = []

for r in range(5):
    for c in range(10):
        estimators.append(ensemble.RandomForestClassifier(n_estimators=n_estimators[r], max_depth=max_features[c]))
# mean_accu = []
cv_generator = LeaveOneGroupOut()
for model in range(len(estimators)):
    cv_score = cross_val_score(estimators[model], train_input, train_data['true_class'], groups=train_data['location_id'],
                               cv=cv_generator)
    print(cv_score)
    save_string = "../saves/cv_forest_model" + str(model) + "_aws_logo.pickle"
    with open(save_string, 'wb') as f:
        pickle.dump(cv_score, f)

print("finished")

# for i in range(len(locations)):
#     test_data = train_df.loc[train_df['location_id'].isin([locations[i]])]
#     train_data = train_df.drop(test_data.index)
#
#     train_input = scaler.transform(train_data[cols])
#     test_input = scaler.transform(test_data[cols])
#
#     for model in range(len(estimators)):
#         clf = estimators[model]
#         clf.fit(train_input, train_data['true_class'])
#         pre = clf.predict(test_input)
#         differ = abs(pre - test_data['true_class'])
#         accu = 1 - np.count_nonzero(differ) / test_data.shape[0]
#         scores[model] += accu
#     print("Fold done")
#     print(scores/(i+1))
#         #print(accu)
#         #mean_accu = mean_accu +accu
#
# print("Average score")
# print(scores/len(locations))
# print("Best")
# print(scores.argmax())
# print("As matrix")
# print(np.reshape(scores, (10, 5)))