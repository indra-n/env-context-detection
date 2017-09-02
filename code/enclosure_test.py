import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import csv

random.seed(42)

# Tests if location is surrounded by walls

#Load the data
fp_indoor_cutsark_in1_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_1'
fp_indoor_cutsark_in2_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_2'
fp_indoor_market_gr1_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_1'
fp_indoor_market_gr2_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_2'
fp_indoor_museum_gr1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_1'
fp_indoor_museum_gr2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_2'
fp_indoor_museum_lw1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\under_light_well\Obs_features_1'
fp_indoor_museum_lw2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\under_light_well\Obs_features_2'

fp_inter_path1_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_1'
fp_inter_path2_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_2'
fp_inter_dept3_sp = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features'
fp_inter_GreenTS_p1_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_1'
fp_inter_GreenTS_p1_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_2'
fp_inter_GreenTS_p2_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_1'
fp_inter_GreenTS_p2_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_2'
fp_inter_market_aw1_sp = '..\data_Greenwich\intermediate\GreenwichMarket\entrance_archway_P1\Obs_features_1'
fp_inter_market_aw2_sp = '..\data_Greenwich\intermediate\GreenwichMarket\entrance_archway_P1\Obs_features_2'
fp_inter_park_dark1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_1'
fp_inter_park_dark2_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_2'
fp_inter_park_light1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_1'
fp_inter_park_light2_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_2'
fp_inter_queens_arch_sp = '..\data_Greenwich\intermediate\QueensHouse\\archway\Obs_features'
fp_inter_queens_col1_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_1'
fp_inter_queens_col2_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_2'

fp_open_park1_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_1'
fp_open_park2_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_2'

fp_urban_sl1_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_1'
fp_urban_sl2_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_2'
fp_urban_cutsark_out1_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_1'
fp_urban_cutsark_out2_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_2'
fp_urban_dept1_sp = '..\data_Greenwich\\urban\Deptford_TrainStation\P1\Obs_features'
fp_urban_dept2_sp = '..\data_Greenwich\\urban\Deptford_TrainStation\P2\Obs_features'
fp_urban_GreenTS_p3_1_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_1'
fp_urban_GreenTS_p3_2_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_2'
fp_urban_queens_court_sp = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features'


# Load in dataframe

#######
# Enclosure labels
# 0 - no enclosure
# 1 - light enclosure (glass walls, open side etc.)
# 2 - enclosing walls

df_indoor_cutsark_in1 = pd.read_csv(fp_indoor_cutsark_in1_sp)
df_indoor_cutsark_in2 = pd.read_csv(fp_indoor_cutsark_in2_sp)
df_indoor_market_gr1 = pd.read_csv(fp_indoor_market_gr1_sp)
df_indoor_market_gr2 = pd.read_csv(fp_indoor_market_gr2_sp)
df_indoor_museum_gr1 = pd.read_csv(fp_indoor_museum_gr1_sp)
df_indoor_museum_gr2 = pd.read_csv(fp_indoor_museum_gr2_sp)
df_indoor_museum_lw1 = pd.read_csv(fp_indoor_museum_lw1_sp)
df_indoor_museum_lw2 = pd.read_csv(fp_indoor_museum_lw2_sp)

df_indoor_cutsark_in1['true_class'] = 1
df_indoor_cutsark_in2['true_class'] = 1
df_indoor_market_gr1['true_class'] = 2
df_indoor_market_gr2['true_class'] = 2
df_indoor_museum_gr1['true_class'] = 2
df_indoor_museum_gr2['true_class'] = 2
df_indoor_museum_lw1['true_class'] = 1
df_indoor_museum_lw2['true_class'] = 1


df_inter_path1 = pd.read_csv(fp_inter_path1_sp)
df_inter_path2 = pd.read_csv(fp_inter_path2_sp)
df_inter_dept3 = pd.read_csv(fp_inter_dept3_sp)
df_inter_GreenTS_p1_1 = pd.read_csv(fp_inter_GreenTS_p1_1_sp)
df_inter_GreenTS_p1_2 = pd.read_csv(fp_inter_GreenTS_p1_2_sp)
df_inter_GreenTS_p2_1 = pd.read_csv(fp_inter_GreenTS_p2_1_sp)
df_inter_GreenTS_p2_2 = pd.read_csv(fp_inter_GreenTS_p2_2_sp)
df_inter_market_aw1 = pd.read_csv(fp_inter_market_aw1_sp)
df_inter_market_aw2 = pd.read_csv(fp_inter_market_aw2_sp)
df_inter_park_dark1 = pd.read_csv(fp_inter_park_dark1_sp)
df_inter_park_dark2 = pd.read_csv(fp_inter_park_dark2_sp)
df_inter_park_light1 = pd.read_csv(fp_inter_park_light1_sp)
df_inter_park_light2 = pd.read_csv(fp_inter_park_light2_sp)
df_inter_queens_arch = pd.read_csv(fp_inter_queens_arch_sp)
df_inter_queens_col1 = pd.read_csv(fp_inter_queens_col1_sp)
df_inter_queens_col2 = pd.read_csv(fp_inter_queens_col2_sp).iloc[:67]

df_inter_path1['true_class'] = 1
df_inter_path2['true_class'] = 1
df_inter_dept3['true_class'] = 1
df_inter_GreenTS_p1_1['true_class'] = 1
df_inter_GreenTS_p1_2['true_class'] = 1
df_inter_GreenTS_p2_1['true_class'] = 1
df_inter_GreenTS_p2_2['true_class'] = 1
df_inter_market_aw1['true_class'] = 1
df_inter_market_aw2['true_class'] = 1
df_inter_park_dark1['true_class'] = 1
df_inter_park_dark2['true_class'] = 1
df_inter_park_light1['true_class'] = 0
df_inter_park_light2['true_class'] = 0
df_inter_queens_arch['true_class'] = 2
df_inter_queens_col1['true_class'] = 1
df_inter_queens_col2['true_class'] = 1


df_open_park1 = pd.read_csv(fp_open_park1_sp)
df_open_park2 = pd.read_csv(fp_open_park2_sp)

df_open_park1['true_class'] = 0
df_open_park2['true_class'] = 0


df_urban_sl1 = pd.read_csv(fp_urban_sl1_sp)
df_urban_sl2 = pd.read_csv(fp_urban_sl2_sp)
df_urban_cutsark_out1 = pd.read_csv(fp_urban_cutsark_out1_sp).iloc[0:38]
df_urban_cutsark_out2 = pd.read_csv(fp_urban_cutsark_out2_sp)
df_urban_dept1 = pd.read_csv(fp_urban_dept1_sp)
df_urban_dept2 = pd.read_csv(fp_urban_dept2_sp)
df_urban_GreenTS_p3_1 = pd.read_csv(fp_urban_GreenTS_p3_1_sp)
df_urban_GreenTS_p3_2 = pd.read_csv(fp_urban_GreenTS_p3_2_sp)
df_urban_queens_court = pd.read_csv(fp_urban_queens_court_sp)


df_urban_sl1['true_class'] = 1
df_urban_sl2['true_class'] = 1
df_urban_cutsark_out1['true_class'] = 0
df_urban_cutsark_out2['true_class'] = 0
df_urban_dept1['true_class'] = 0
df_urban_dept2['true_class'] = 0
df_urban_GreenTS_p3_1['true_class'] = 0
df_urban_GreenTS_p3_2['true_class'] = 0
df_urban_queens_court['true_class'] = 2

#cols = ['obs_id', 'e_id', 'sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
# cols = ['sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'elev_0_30', 'elev_30_60', 'elev_60_90',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
# cols=['num_sat_25', 'sum_snr_25',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'elev_0_30',
         'elev_0_30_25']


#######
# Location values
df_indoor_cutsark_in1['location'] = 321
df_indoor_cutsark_in2['location'] = 322
df_indoor_market_gr1['location'] = 421
df_indoor_market_gr2['location'] = 422
df_indoor_museum_gr1['location'] = 511
df_indoor_museum_gr2['location'] = 512
df_indoor_museum_lw1['location'] = 521
df_indoor_museum_lw2['location'] = 522

df_inter_path1['location'] = 211
df_inter_path2['location'] = 212
df_inter_dept3['location'] = 931
df_inter_GreenTS_p1_1['location'] = 811
df_inter_GreenTS_p1_2['location'] = 812
df_inter_GreenTS_p2_1['location'] = 821
df_inter_GreenTS_p2_2['location'] = 822
df_inter_market_aw1['location'] = 411
df_inter_market_aw2['location'] = 412
df_inter_park_dark1['location'] = 721
df_inter_park_dark2['location'] = 722
df_inter_park_light1['location'] = 731
df_inter_park_light2['location'] = 732
df_inter_queens_arch['location'] = 631
df_inter_queens_col1['location'] = 611
df_inter_queens_col2['location'] = 612

df_open_park1['location'] = 711
df_open_park2['location'] = 712

df_urban_sl1['location'] = 111
df_urban_sl2['location'] = 112
df_urban_cutsark_out1['location'] = 311
df_urban_cutsark_out2['location'] = 312
df_urban_dept1['location'] = 911
df_urban_dept2['location'] = 921
df_urban_GreenTS_p3_1['location'] = 831
df_urban_GreenTS_p3_2['location'] = 832
df_urban_queens_court['location'] = 621


#######
# Alternative assignments
# 1- indoor
# 2- inbetween
# 3- urban
# 4- open sky
# 5- i don't know

# df_indoor_cutsark_in1['true_class'] = 1
# df_indoor_cutsark_in2['true_class'] = 1
# df_indoor_market_gr1['true_class'] = 1
# df_indoor_market_gr2['true_class'] = 1
# df_indoor_museum_gr1['true_class'] = 1
# df_indoor_museum_gr2['true_class'] = 1
# df_indoor_museum_lw1['true_class'] = 2
# df_indoor_museum_lw2['true_class'] = 2
# # df_indoor_cutsark_in1['true_class'] = 5
# # df_indoor_cutsark_in2['true_class'] = 5
# # df_indoor_market_gr1['true_class'] = 5
# # df_indoor_market_gr2['true_class'] = 5
# # df_indoor_museum_gr1['true_class'] = 5
# # df_indoor_museum_gr2['true_class'] = 5
# # df_indoor_museum_lw1['true_class'] = 5
# # df_indoor_museum_lw2['true_class'] = 5
#
# df_inter_path1['true_class'] = 3
# df_inter_path2['true_class'] = 3
# df_inter_dept3['true_class'] = 3
# df_inter_GreenTS_p1_1['true_class'] = 3
# df_inter_GreenTS_p1_2['true_class'] = 3
# df_inter_GreenTS_p2_1['true_class'] = 3
# df_inter_GreenTS_p2_2['true_class'] = 3
# df_inter_market_aw1['true_class'] = 2
# df_inter_market_aw2['true_class'] = 2
# # df_inter_market_aw1['true_class'] = 5
# # df_inter_market_aw2['true_class'] = 5
#
# df_inter_park_dark1['true_class'] = 2
# df_inter_park_dark2['true_class'] = 2
# # df_inter_park_dark1['true_class'] = 5
# # df_inter_park_dark2['true_class'] = 5
#
# df_inter_park_light1['true_class'] = 3
# df_inter_park_light2['true_class'] = 3
#
# df_inter_queens_arch['true_class'] = 2
# #df_inter_queens_arch['true_class'] = 5
#
# df_inter_queens_col1['true_class'] = 3
# df_inter_queens_col2['true_class'] = 3
#
# df_urban_sl1['true_class'] = 3
# df_urban_sl2['true_class'] = 3
# df_urban_cutsark_out1['true_class'] = 3
# df_urban_cutsark_out2['true_class'] = 3
# df_urban_dept1['true_class'] = 4
# df_urban_dept2['true_class'] = 3
# df_urban_GreenTS_p3_1['true_class'] = 3
# df_urban_GreenTS_p3_2['true_class'] = 3
# df_urban_queens_court['true_class'] = 2
#
# df_open_park1['true_class'] = 4
# df_open_park2['true_class'] = 4



# Split training and test data


df_indoor_cutsark_in = pd.concat([df_indoor_cutsark_in1, df_indoor_cutsark_in2])
train_indoor_1 = df_indoor_cutsark_in.sample(60)
test_indoor_1 = df_indoor_cutsark_in.drop(train_indoor_1.index).sample(60)

df_indoor_market_gr = pd.concat([df_indoor_market_gr1, df_indoor_market_gr2])
train_indoor_2 = df_indoor_market_gr.sample(40)
test_indoor_2 = df_indoor_market_gr.drop(train_indoor_2.index).sample(60)

df_indoor_museum_gr = pd.concat([df_indoor_museum_gr1, df_indoor_museum_gr2])
train_indoor_3 = df_indoor_museum_gr.sample(60)
test_indoor_3 = df_indoor_museum_gr2.drop(train_indoor_3.index).sample(60)


train_indoor_4 = df_indoor_museum_lw1.sample(30)
test_indoor_4 = df_indoor_museum_lw2.sample(15)

df_inter_path = pd.concat([df_inter_path1, df_inter_path2])
train_inter_1 = df_inter_path.sample(40)
test_inter_1 = df_inter_path.drop(train_inter_1.index).sample(60)


test_inter_2 = df_inter_dept3.sample(60)

df_inter_GreenTS_p1 = pd.concat([df_inter_GreenTS_p1_1, df_inter_GreenTS_p1_2])
train_inter_2 = df_inter_GreenTS_p1.sample(60)
test_inter_3 = df_inter_GreenTS_p1.drop(train_inter_2.index).sample(60)

train_inter_3 = df_inter_GreenTS_p2_1.sample(60)
test_inter_4 = df_inter_GreenTS_p2_2.sample(60)
train_inter_4 = df_inter_market_aw1.sample(40)
test_inter_5 = df_inter_market_aw2.sample(60)
train_inter_5 = df_inter_park_dark1.sample(40)
test_inter_6 = df_inter_park_dark2.sample(60)
train_inter_6 = df_inter_park_light1.sample(60)
test_inter_9 = df_inter_park_light2.sample(60)
test_inter_7 = df_inter_queens_arch.sample(60)
train_inter_7 = df_inter_queens_col1.sample(60)
test_inter_8 = df_inter_queens_col2.sample(60)

df_urban_sl = pd.concat([df_urban_sl1, df_urban_sl2])
train_urban_1 = df_urban_sl.sample(60)
test_urban_1 = df_urban_sl.drop(train_urban_1.index).sample(60)

df_urban_cutsark_out = pd.concat([df_urban_cutsark_out1, df_urban_cutsark_out2])
train_urban_2 = df_urban_cutsark_out.sample(50)
test_urban_2 = df_urban_cutsark_out2.drop(train_urban_2.index).sample(50)

train_urban_3 = df_urban_dept1.sample(60)
test_urban_3 = df_urban_dept2.sample(60)

train_urban_4 = df_urban_GreenTS_p3_1.sample(40)
test_urban_4 = df_urban_GreenTS_p3_2.sample(60)
train_urban_5 = df_urban_queens_court.sample(60)

train_open = df_open_park1.sample(60)
test_open = df_open_park2.sample(60)

#########
# train_indoor_bm = df_indoor_bm.sample(100)
# train_indoor_ch2221 = df_indoor_ch2221.sample(100)
# train_indoor_ch103a = df_indoor_ch103a.sample(100)
# train_indoor_jah = df_indoor_jah.sample(100)
#
# test_indoor_bm = df_indoor_bm.drop(train_indoor_bm.index).sample(100)
# test_indoor_ch2221 = df_indoor_ch2221.drop(train_indoor_ch2221.index).sample(100)
# test_indoor_ch103a = df_indoor_ch103a.drop(train_indoor_ch103a.index).sample(100)
# test_indoor_ch103b = df_indoor_ch103b.sample(100)
# test_indoor_jah = df_indoor_jah.drop(train_indoor_jah.index).sample(100)
#
# train_inter = df_inter.sample(100)
# test_inter = df_inter.drop(train_inter.index).sample(100)
#
# train_urban_p1b = df_urban_p1b.sample(100)
# train_urban_p2b = df_urban_p2b.sample(100)
# train_urban_p4b = df_urban_p4b.sample(100)
#
# test_urban_p1b = df_urban_p1b.drop(train_urban_p1b.index).sample(100)
# test_urban_p2b = df_urban_p2b.drop(train_urban_p2b.index).sample(100)
# test_urban_p3b = df_urban_p3b.sample(100)
# test_urban_p4b = df_urban_p4b.drop(train_urban_p4b.index).sample(100)
#
# train_open_reg = df_open_reg.sample(100)
# test_open_hyde = df_open_hyde.sample(100)

# train_df = [train_indoor_bm, train_indoor_ch2221, train_indoor_ch103a, train_indoor_jah, train_inter, train_urban_p1b,
#             train_urban_p2b, train_urban_p4b, train_open_reg]

train_df = [train_indoor_2, train_indoor_3, train_indoor_4, train_inter_1, train_inter_2, train_inter_3,
            train_inter_4, train_inter_5, train_inter_6, train_inter_7, train_urban_1, train_urban_2, train_urban_3,
            train_urban_4, train_urban_5, train_open, test_urban_3]

# train_df = [train_indoor_1, train_indoor_2, train_indoor_3, train_indoor_4, train_inter_1, train_inter_2, train_inter_3,
#             train_inter_4, train_inter_5, train_inter_6, train_inter_7, train_urban_1, train_urban_2, train_urban_3,
#             train_urban_4, train_urban_5, train_open]
train_data = pd.concat(train_df).sample(frac=1).reset_index(drop=True)

# test_df = [test_indoor_bm, test_indoor_ch2221, test_indoor_ch103a, test_indoor_ch103b, test_indoor_jah, test_inter,
#            test_urban_p1b, test_urban_p2b, test_urban_p3b, test_urban_p4b, test_open_hyde]
test_df = [test_indoor_2, test_indoor_3, test_indoor_4, test_inter_1, test_inter_2, test_inter_3,
           test_inter_4, test_inter_5, test_inter_6, test_inter_7, test_inter_8, test_inter_9, test_urban_1,
           test_urban_2, test_urban_3, test_urban_4, test_open]
# test_df = [test_indoor_1, test_indoor_2, test_indoor_3, test_indoor_4, test_inter_1, test_inter_2, test_inter_3,
#            test_inter_4, test_inter_5, test_inter_6, test_inter_7, test_inter_8, test_inter_9, test_urban_1,
#            test_urban_2, test_urban_3, test_urban_4, test_open]
test_data = pd.concat(test_df).sample(frac=1).reset_index(drop=True)

forest = ensemble.RandomForestClassifier(n_estimators=100)
forest.fit(train_data[cols], train_data['true_class'])

pred = forest.predict(test_data[cols])
pred_probas = forest.predict_proba(test_data[cols])

pred_probas_dept = forest.predict_proba(test_inter_7[cols])
pred_dept = forest.predict(test_inter_7[cols])
differ_dept = abs(pred_dept - test_inter_7['true_class'])
accu_dept = 1 - np.count_nonzero(differ_dept) / test_inter_7.shape[0]

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]
print(accu)
print(accu_dept)

wrong_pred = test_data[differ != 0]
print(wrong_pred.shape)
print(wrong_pred['location'].value_counts())

cm = confusion_matrix(test_data['true_class'], pred)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((3, 1))
print(cm_proc)

# print(pred_probas_dept)
# for i in range(1000):
#     if differ[i] != 0:
#         print(test_data['true_class'][i])
#         print(pred_probas[i])

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(test_data[cols].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(test_data[cols].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(test_data[cols].shape[1]), indices)
plt.xlim([-1, test_data[cols].shape[1]])
plt.show()