import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler

# cols = ['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'elev_0_30', 'elev_30_60', 'elev_60_90',
#         'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']

cols = ['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'num_sat_u20', 'sum_sat_u20', 'elev_0_30', 'elev_30_60',
'elev_60_90', 'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90', 'elev_0_30_25', 'elev_30_60_25',
'elev_60_90_25', 'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25', 'elev_0_30_weak', 'elev_30_60_weak',
 'elev_60_90_weak', 'sum_elev_0_30_weak', 'sum_elev_30_60_weak', 'sum_elev_60_90_weak']


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

train_data = train_df
test_data = test_df


#########
# Experiment with Han's data
# with open('..\saves\given_data_sample_full.pickle', 'rb') as f:
#     given_data = pickle.load(f)
#

# test_data = given_data

#######


print(train_data['true_class'].value_counts())

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])


forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3)
forest.fit(train_input, train_data['true_class'])

pred = forest.predict(test_input)
pred_probas = forest.predict_proba(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print(accu)

cm = confusion_matrix(test_data['true_class'], pred)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(test_data[cols].shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], cols[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(test_data[cols].shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(test_data[cols].shape[1]), indices)
plt.xlim([-1, test_data[cols].shape[1]])
plt.show()

# with open('..\saves\\tree_model1.pickle', 'wb') as f:
#     pickle.dump(forest, f)
