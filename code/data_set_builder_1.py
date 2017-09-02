import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the data

fp_indoor_cutsark_in_1 = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_ext_4'
fp_indoor_coffee_1 = '..\data_Greenwich\indoor\CoffeeShop\Obs_features_ext_4'
fp_indoor_station_1 = '..\data_Greenwich\indoor\Greenwich_TrainStation\Obs_features_ext_4'
fp_indoor_market_gr_1 = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_ext_4'
fp_indoor_museum_gr_1 = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_ext_4'
fp_indoor_museum_win_1 = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\Obs_features_ext_4'

fp_inter_path_1 = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_ext_4'
fp_inter_dept3_1 = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features_ext_4'
fp_inter_GreenTS_p1_1 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_ext_4'
fp_inter_GreenTS_p2_1 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_ext_4'
fp_inter_park_light_1 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_ext_4'
fp_inter_uni_col1_1 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\Obs_features_ext_4'
fp_inter_uni_col2_1 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\Obs_features_ext_4'
fp_inter_queens_col_1 = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_ext_4'

fp_open_park_1 = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_ext_4'
fp_open_park_cric_1 = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\Obs_features_ext_4'
fp_open_blackheath_1 = '..\data_Greenwich\open_sky\Blackheath\Obs_features_ext_4'
fp_open_park_view_1 = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\Obs_features_ext_4'
fp_open_rangers_1 = '..\data_Greenwich\open_sky\RangersHouse\Obs_features_ext_4'
fp_open_hill_1 = '..\data_Greenwich\open_sky\DartmouthHill\Obs_features_ext_4'

fp_urban_sl_1 = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_ext_4'
fp_urban_cutsark_out_1 = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_ext_4'
fp_urban_tesco_1 = '..\data_Greenwich\\urban\ByTesco\Obs_features_ext_4'
fp_urban_GreenTS_p3_1 = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_ext_4'
fp_urban_uni_wall_1 = '..\data_Greenwich\\urban\GreenwichUni\Obs_features_ext_4'
fp_urban_meridian_1 = '..\data_Greenwich\\urban\MeridianPassage\Obs_features_ext_4'
fp_urban_queens_court_1 = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features_ext_4'
fp_urban_park_trees_1 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_ext_4'

######

fp_indoor_cutsark_in_2 = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_ext_5'
fp_indoor_coffee_2 = '..\data_Greenwich\indoor\CoffeeShop\Obs_features_ext_5'
fp_indoor_station_2 = '..\data_Greenwich\indoor\Greenwich_TrainStation\Obs_features_ext_5'
fp_indoor_market_gr_2 = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_ext_5'
fp_indoor_museum_gr_2 = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_ext_5'
fp_indoor_museum_win_2 = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\Obs_features_ext_5'

fp_inter_path_2 = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_ext_5'
fp_inter_dept3_2 = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features_ext_5'
fp_inter_GreenTS_p1_2 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_ext_5'
fp_inter_GreenTS_p2_2 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_ext_5'
fp_inter_uni_col1_2 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\Obs_features_ext_5'
fp_inter_uni_col2_2 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\Obs_features_ext_5'
fp_inter_queens_col_2 = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_ext_5'

fp_open_park_2 = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_ext_5'
fp_open_park_cric_2 = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\Obs_features_ext_5'
fp_open_blackheath_2 = '..\data_Greenwich\open_sky\Blackheath\Obs_features_ext_5'
fp_open_park_view_2 = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\Obs_features_ext_5'
fp_open_rangers_2 = '..\data_Greenwich\open_sky\RangersHouse\Obs_features_ext_5'
fp_open_hill_2 = '..\data_Greenwich\open_sky\DartmouthHill\Obs_features_ext_5'

fp_urban_sl_2 = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_ext_5'
fp_urban_cutsark_out_2 = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_ext_5'
fp_urban_tesco_2 = '..\data_Greenwich\\urban\ByTesco\Obs_features_ext_5'
fp_urban_GreenTS_p3_2 = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_ext_5'
fp_urban_uni_wall_2 = '..\data_Greenwich\\urban\GreenwichUni\Obs_features_ext_5'
fp_urban_meridian_2 = '..\data_Greenwich\\urban\MeridianPassage\Obs_features_ext_5'
fp_urban_queens_court_2 = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features_ext_5'
fp_urban_park_trees_2 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_ext_5'

df_indoor_cutsark_in_1 = pd.read_csv(fp_indoor_cutsark_in_1).iloc[:260]  # consistently a lot of satelites
df_indoor_coffee_1 = pd.read_csv(fp_indoor_coffee_1)  # typical indoors
df_indoor_station_1 = pd.read_csv(fp_indoor_station_1).iloc[:300]  # have many points with high numbers (open doors)
df_indoor_market_gr_1 = pd.read_csv(fp_indoor_market_gr_1).iloc[:250] # consistently high and hard to identify
df_indoor_museum_gr_1 = pd.read_csv(fp_indoor_museum_gr_1)  # typical indoor
df_indoor_museum_win_1 = pd.read_csv(fp_indoor_museum_win_1).iloc[:250]  # quite a lot of satelites

df_inter_path_1 = pd.read_csv(fp_inter_path_1).iloc[40:]
df_inter_dept3_1 = pd.read_csv(fp_inter_dept3_1).iloc[:250]
df_inter_GreenTS_p1_1 = pd.read_csv(fp_inter_GreenTS_p1_1).iloc[20:]
df_inter_GreenTS_p2_1 = pd.read_csv(fp_inter_GreenTS_p2_1)
df_inter_park_light_1 = pd.read_csv(fp_inter_park_light_1)
df_inter_uni_col1_1 = pd.read_csv(fp_inter_uni_col1_1).iloc[:315]
df_inter_uni_col2_1 = pd.read_csv(fp_inter_uni_col2_1).iloc[:250]  # a looot of satelites
df_inter_queens_col_1 = pd.read_csv(fp_inter_queens_col_1).iloc[:350]

df_open_park_1 = pd.read_csv(fp_open_park_1).iloc[:228]
df_open_park_cric_1 = pd.read_csv(fp_open_park_cric_1).iloc[:190]
df_open_blackheath_1 = pd.read_csv(fp_open_blackheath_1).iloc[:275]
df_open_GreenTS_p3_1 = pd.read_csv(fp_urban_GreenTS_p3_1)
df_open_park_view_1 = pd.read_csv(fp_open_park_view_1)
df_open_rangers_1 = pd.read_csv(fp_open_rangers_1)
# df_open_hill_1 = pd.read_csv(fp_open_hill_1)

df_urban_sl_1 = pd.read_csv(fp_urban_sl_1).iloc[:240]
df_urban_cutsark_out_1 = pd.read_csv(fp_urban_cutsark_out_1).iloc[:280]
df_urban_tesco_1 = pd.read_csv(fp_urban_tesco_1).iloc[:220]
df_urban_uni_wall_1 = pd.read_csv(fp_urban_uni_wall_1).iloc[10:260]
df_urban_meridian_1 = pd.read_csv(fp_urban_meridian_1)
df_urban_queens_court_1 = pd.read_csv(fp_urban_queens_court_1).iloc[:300]
df_urban_park_trees_1 = pd.read_csv(fp_urban_park_trees_1)

########

df_indoor_cutsark_in_2 = pd.read_csv(fp_indoor_cutsark_in_2).iloc[:270]
df_indoor_coffee_2 = pd.read_csv(fp_indoor_coffee_2)
df_indoor_station_2 = pd.read_csv(fp_indoor_station_2).iloc[:280]
df_indoor_market_gr_2 = pd.read_csv(fp_indoor_market_gr_2).iloc[:340]
df_indoor_museum_gr_2 = pd.read_csv(fp_indoor_museum_gr_2)
df_indoor_museum_win_2 = pd.read_csv(fp_indoor_museum_win_2).iloc[:310]

df_inter_path_2 = pd.read_csv(fp_inter_path_2)
df_inter_dept3_2 = pd.read_csv(fp_inter_dept3_2).iloc[:260]
df_inter_GreenTS_p1_2 = pd.read_csv(fp_inter_GreenTS_p1_2)
df_inter_GreenTS_p2_2 = pd.read_csv(fp_inter_GreenTS_p2_2).iloc[:220]
df_inter_uni_col1_2 = pd.read_csv(fp_inter_uni_col1_2).iloc[:325]
df_inter_uni_col2_2 = pd.read_csv(fp_inter_uni_col2_2)
df_inter_queens_col_2 = pd.read_csv(fp_inter_queens_col_2).iloc[:310]

df_open_park_2 = pd.read_csv(fp_open_park_2).iloc[:300]
df_open_park_cric_2 = pd.read_csv(fp_open_park_cric_2).iloc[:220]
df_open_blackheath_2 = pd.read_csv(fp_open_blackheath_2).iloc[:310]
df_open_GreenTS_p3_2 = pd.read_csv(fp_urban_GreenTS_p3_2).iloc[:270]
df_open_park_view_2 = pd.read_csv(fp_open_park_view_2).iloc[:250]
df_open_rangers_2 = pd.read_csv(fp_open_rangers_2).iloc[:300]
df_open_hill_2 = pd.read_csv(fp_open_hill_2).iloc[:250]

df_urban_sl_2 = pd.read_csv(fp_urban_sl_2)
df_urban_cutsark_out_2 = pd.read_csv(fp_urban_cutsark_out_2).iloc[:240]
df_urban_tesco_2 = pd.read_csv(fp_urban_tesco_2).iloc[50:335]
df_urban_uni_wall_2 = pd.read_csv(fp_urban_uni_wall_2).iloc[:355]
df_urban_meridian_2 = pd.read_csv(fp_urban_meridian_2)
df_urban_queens_court_2 = pd.read_csv(fp_urban_queens_court_2).iloc[:295]
df_urban_park_trees_2 = pd.read_csv(fp_urban_park_trees_2)

# Print the size of sets
# print(df_indoor_cutsark_in_1.shape)
# print(df_indoor_coffee_1.shape)
# print(df_indoor_station_1.shape)
# print(df_indoor_market_gr_1.shape)
# print(df_indoor_museum_gr_1.shape)
# print(df_indoor_museum_win_1.shape)
#
# print(df_inter_path_1.shape)
# print(df_inter_dept3_1.shape)
# print(df_inter_GreenTS_p1_1.shape)
# print(df_inter_GreenTS_p2_1.shape)
# print(df_urban_park_trees_1.shape)
# print(df_inter_park_light_1.shape)
# print(df_inter_uni_col1_1.shape)
# print(df_inter_uni_col2_1.shape)
# print(df_inter_queens_col_1.shape)
#
# print(df_open_park_1.shape)
# print(df_open_park_cric_1.shape)
# print(df_open_blackheath_1.shape)
#
# print(df_urban_sl_1.shape)
# print(df_urban_cutsark_out_1.shape)
# print(df_urban_tesco_1.shape)
# print(df_open_GreenTS_p3_1.shape)
# print(df_urban_uni_wall_1.shape)
# print(df_urban_meridian_1.shape)
# print(df_urban_queens_court_1.shape)

#####

# print(df_indoor_cutsark_in_2.shape)
# print(df_indoor_coffee_2.shape)
# print(df_indoor_station_2.shape)
# print(df_indoor_market_gr_2.shape)
# print(df_indoor_museum_gr_2.shape)
# print(df_indoor_museum_win_2.shape)
#
# print(df_inter_path_2.shape)
# print(df_inter_dept3_2.shape)
# print(df_inter_GreenTS_p1_2.shape)
# print(df_inter_GreenTS_p2_2.shape)
# print(df_urban_park_trees_2.shape)
# print(df_inter_uni_col1_2.shape)
# print(df_inter_uni_col2_2.shape)
# print(df_inter_queens_col_2.shape)
#
# print(df_open_park_2.shape)
# print(df_open_park_cric_2.shape)
# print(df_open_blackheath_2.shape)
#
# print(df_urban_sl_2.shape)
# print(df_urban_cutsark_out_2.shape)
# print(df_urban_tesco_2.shape)
# print(df_open_GreenTS_p3_2.shape)
# print(df_urban_uni_wall_2.shape)
# print(df_urban_meridian_2.shape)
# print(df_urban_queens_court_2.shape)

df_indoor_cutsark_in_1['true_class'] = 1
df_indoor_coffee_1['true_class'] = 1
df_indoor_station_1['true_class'] = 1
df_indoor_market_gr_1['true_class'] = 1
df_indoor_museum_gr_1['true_class'] = 1
df_indoor_museum_win_1['true_class'] = 1

df_inter_path_1['true_class'] = 2
df_inter_dept3_1['true_class'] = 2
df_inter_GreenTS_p1_1['true_class'] = 2
df_inter_GreenTS_p2_1['true_class'] = 2
df_inter_park_light_1['true_class'] = 2
df_inter_uni_col1_1['true_class'] = 2
df_inter_uni_col2_1['true_class'] = 2
df_inter_queens_col_1['true_class'] = 2

df_urban_sl_1['true_class'] = 3
df_urban_cutsark_out_1['true_class'] = 3
df_urban_tesco_1['true_class'] = 3
df_urban_uni_wall_1['true_class'] = 3
df_urban_meridian_1['true_class'] = 3
df_urban_queens_court_1['true_class'] = 3
df_urban_park_trees_1['true_class'] = 3

df_open_park_1['true_class'] = 4
df_open_park_cric_1['true_class'] = 4
df_open_blackheath_1['true_class'] = 4
df_open_GreenTS_p3_1['true_class'] = 4
df_open_park_view_1['true_class'] = 4
df_open_rangers_1['true_class'] = 4
# df_open_hill_1['true_class'] = 4

df_indoor_cutsark_in_1['location_id'] = 324
df_indoor_coffee_1['location_id'] = 1414
df_indoor_station_1['location_id'] = 844
df_indoor_market_gr_1['location_id'] = 424
df_indoor_museum_gr_1['location_id'] = 514
df_indoor_museum_win_1['location_id'] = 534

df_inter_path_1['location_id'] = 214
df_inter_dept3_1['location_id'] = 934
df_inter_GreenTS_p1_1['location_id'] = 814
df_inter_GreenTS_p2_1['location_id'] = 824
df_inter_park_light_1['location_id'] = 734
df_inter_uni_col1_1['location_id'] = 1114
df_inter_uni_col2_1['location_id'] = 1124
df_inter_queens_col_1['location_id'] = 614

df_urban_sl_1['location_id'] = 114
df_urban_cutsark_out_1['location_id'] = 314
df_urban_tesco_1['location_id'] = 1214
df_urban_uni_wall_1['location_id'] = 1134
df_urban_meridian_1['location_id'] = 1314
df_urban_queens_court_1['location_id'] = 624
df_urban_park_trees_1['location_id'] = 724

df_open_park_1['location_id'] = 714
df_open_park_cric_1['location_id'] = 744
df_open_blackheath_1['location_id'] = 1014
df_open_GreenTS_p3_1['location_id'] = 834
df_open_park_view_1['location_id'] = 754
df_open_rangers_1['location_id'] = 1024
# df_open_hill_1['location_id'] = 1034

########

df_indoor_cutsark_in_2['true_class'] = 1
df_indoor_coffee_2['true_class'] = 1
df_indoor_station_2['true_class'] = 1
df_indoor_market_gr_2['true_class'] = 1
df_indoor_museum_gr_2['true_class'] = 1
df_indoor_museum_win_2['true_class'] = 1

df_inter_path_2['true_class'] = 2
df_inter_dept3_2['true_class'] = 2
df_inter_GreenTS_p1_2['true_class'] = 2
df_inter_GreenTS_p2_2['true_class'] = 2
df_inter_uni_col1_2['true_class'] = 2
df_inter_uni_col2_2['true_class'] = 2
df_inter_queens_col_2['true_class'] = 2

df_urban_sl_2['true_class'] = 3
df_urban_cutsark_out_2['true_class'] = 3
df_urban_tesco_2['true_class'] = 3
df_urban_uni_wall_2['true_class'] = 3
df_urban_meridian_2['true_class'] = 3
df_urban_queens_court_2['true_class'] = 3
df_urban_park_trees_2['true_class'] = 3

df_open_park_2['true_class'] = 4
df_open_park_cric_2['true_class'] = 4
df_open_blackheath_2['true_class'] = 4
df_open_GreenTS_p3_2['true_class'] = 4
df_open_park_view_2['true_class'] = 4
df_open_rangers_2['true_class'] = 4
df_open_hill_2['true_class'] = 4

df_indoor_cutsark_in_2['location_id'] = 325
df_indoor_coffee_2['location_id'] = 1415
df_indoor_station_2['location_id'] = 845
df_indoor_market_gr_2['location_id'] = 425
df_indoor_museum_gr_2['location_id'] = 515
df_indoor_museum_win_2['location_id'] = 535

df_inter_path_2['location_id'] = 215
df_inter_dept3_2['location_id'] = 935
df_inter_GreenTS_p1_2['location_id'] = 815
df_inter_GreenTS_p2_2['location_id'] = 825
df_inter_uni_col1_2['location_id'] = 1115
df_inter_uni_col2_2['location_id'] = 1125
df_inter_queens_col_2['location_id'] = 615

df_urban_sl_2['location_id'] = 115
df_urban_cutsark_out_2['location_id'] = 315
df_urban_tesco_2['location_id'] = 1215
df_urban_uni_wall_2['location_id'] = 1135
df_urban_meridian_2['location_id'] = 1315
df_urban_queens_court_2['location_id'] = 625
df_urban_park_trees_2['location_id'] = 725

df_open_park_2['location_id'] = 715
df_open_park_cric_2['location_id'] = 745
df_open_blackheath_2['location_id'] = 1015
df_open_GreenTS_p3_2['location_id'] = 835
df_open_park_view_2['location_id'] = 755
df_open_rangers_2['location_id'] = 1025
df_open_hill_2['location_id'] = 1035

train_indoor_df = [df_indoor_cutsark_in_1, df_indoor_coffee_1, df_indoor_station_1,
                   df_indoor_museum_gr_1, df_indoor_museum_win_1]
# train_indoor = pd.concat(train_indoor_df).sample(1500).reset_index(drop=True)
train_indoor = pd.concat(train_indoor_df).reset_index(drop=True)

train_inter_df = [df_inter_path_1, df_inter_GreenTS_p1_1,
                  df_inter_uni_col1_1, df_inter_uni_col2_1, df_inter_queens_col_1]
# train_inter = pd.concat(train_inter_df).sample(1500).reset_index(drop=True)
train_inter = pd.concat(train_inter_df).reset_index(drop=True)

train_urban_df = [df_urban_tesco_1, df_urban_uni_wall_1,
                  df_urban_meridian_1, df_urban_queens_court_1, df_urban_park_trees_1]
# train_urban = pd.concat(train_urban_df).sample(1500).reset_index(drop=True)
train_urban = pd.concat(train_urban_df).reset_index(drop=True)

train_open_df = [df_open_park_cric_1, df_open_blackheath_1, df_open_park_view_1, df_open_rangers_1]
# train_open = pd.concat(train_open_df).sample(900).reset_index(drop=True)
train_open = pd.concat(train_open_df).reset_index(drop=True)

greenwich_train_df = [train_indoor, train_inter, train_urban, train_open]

test_indoor_df = [df_indoor_cutsark_in_2, df_indoor_coffee_2, df_indoor_station_2,
                   df_indoor_museum_gr_2, df_indoor_museum_win_2]
# test_indoor = pd.concat(test_indoor_df).sample(1500).reset_index(drop=True)
test_indoor = pd.concat(test_indoor_df).reset_index(drop=True)

test_inter_df = [df_inter_path_2, df_inter_GreenTS_p1_2,
                  df_inter_uni_col1_2, df_inter_uni_col2_2, df_inter_queens_col_2]
# test_inter = pd.concat(test_inter_df).sample(1500).reset_index(drop=True)
test_inter = pd.concat(test_inter_df).reset_index(drop=True)

test_urban_df = [df_urban_tesco_2, df_urban_uni_wall_2,
                  df_urban_meridian_2, df_urban_queens_court_2, df_urban_park_trees_2]
# test_urban = pd.concat(test_urban_df).sample(1500).reset_index(drop=True)
test_urban = pd.concat(test_urban_df).reset_index(drop=True)

test_open_df = [df_open_park_cric_2, df_open_blackheath_2, df_open_park_view_2, df_open_rangers_2]
# test_open = pd.concat(test_open_df).sample(900).reset_index(drop=True)
test_open = pd.concat(test_open_df).reset_index(drop=True)

greenwich_test_df = [test_indoor, test_inter, test_urban, test_open]


greenwich_train_data = pd.concat(greenwich_train_df).sample(frac=1).reset_index(drop=True)
greenwich_test_data = pd.concat(greenwich_test_df).sample(frac=1).reset_index(drop=True)

print(greenwich_train_data['true_class'].value_counts())
print(greenwich_test_data['true_class'].value_counts())


greenwich_data = greenwich_test_data
greenwich_data = df_inter_uni_col2_2
sorted_df = greenwich_data.sort_values(by='num_sat_25', ascending=1)
print(sorted_df.head(n=50)[['e_id', 'num_sat_25', 'sum_snr_25']])

greenwich_train_data['num_sat_weak'] = greenwich_train_data['num_sat'] - greenwich_train_data['num_sat_25']
greenwich_train_data['sum_snr_weak'] = greenwich_train_data['sum_snr'] - greenwich_train_data['sum_snr_25']
greenwich_test_data['num_sat_weak'] = greenwich_test_data['num_sat'] - greenwich_test_data['num_sat_25']
greenwich_test_data['sum_snr_weak'] = greenwich_test_data['sum_snr'] - greenwich_test_data['sum_snr_25']

greenwich_train_data['elev_0_30_weak'] = greenwich_train_data['elev_0_30'] - greenwich_train_data['elev_0_30_25']
greenwich_train_data['sum_elev_0_30_weak'] = greenwich_train_data['sum_elev_0_30'] - greenwich_train_data['sum_elev_0_30_25']
greenwich_test_data['elev_0_30_weak'] = greenwich_test_data['elev_0_30'] - greenwich_test_data['elev_0_30_25']
greenwich_test_data['sum_elev_0_30_weak'] = greenwich_test_data['sum_elev_0_30'] - greenwich_test_data['sum_elev_0_30_25']

greenwich_train_data['elev_30_60_weak'] = greenwich_train_data['elev_30_60'] - greenwich_train_data['elev_30_60_25']
greenwich_train_data['sum_elev_30_60_weak'] = greenwich_train_data['sum_elev_30_60'] - greenwich_train_data['sum_elev_30_60_25']
greenwich_test_data['elev_30_60_weak'] = greenwich_test_data['elev_30_60'] - greenwich_test_data['elev_30_60_25']
greenwich_test_data['sum_elev_30_60_weak'] = greenwich_test_data['sum_elev_30_60'] - greenwich_test_data['sum_elev_30_60_25']

greenwich_train_data['elev_60_90_weak'] = greenwich_train_data['elev_60_90'] - greenwich_train_data['elev_60_90_25']
greenwich_train_data['sum_elev_60_90_weak'] = greenwich_train_data['sum_elev_60_90'] - greenwich_train_data['sum_elev_60_90_25']
greenwich_test_data['elev_60_90_weak'] = greenwich_test_data['elev_60_90'] - greenwich_test_data['elev_60_90_25']
greenwich_test_data['sum_elev_60_90_weak'] = greenwich_test_data['sum_elev_60_90'] - greenwich_test_data['sum_elev_60_90_25']


with open('..\saves\\exp_train_trim_1.pickle', 'wb') as f:
    pickle.dump(greenwich_train_data, f)

with open('..\saves\\exp_test_trim_1.pickle', 'wb') as f:
    pickle.dump(greenwich_test_data, f)

# #########
plot_data_full = greenwich_data
cols = ['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'snr_std',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25', 'frac_30']

plot_data = np.column_stack([np.log(1 + plot_data_full['num_sat_25']), np.log(1 + plot_data_full['sum_snr_25']),
                             plot_data_full['true_class']])

print(plot_data.shape)

idx_1 = plot_data[:, 2] == 1
idx_2 = plot_data[:, 2] == 2
idx_3 = plot_data[:, 2] == 3
idx_4 = plot_data[:, 2] == 4

plt.xlabel('Log(# satelites with SNR over 25Hz)', fontsize=10)
plt.ylabel('Log(sum of SNR for satelites with SNR over 25Hz)', fontsize=10)
plt.scatter(plot_data[idx_1, 0], plot_data[idx_1, 1], label="indoor", c='r', s=8)
plt.scatter(plot_data[idx_2, 0], plot_data[idx_2, 1], label="inter", c='b', s=8)
plt.scatter(plot_data[idx_3, 0], plot_data[idx_3, 1], label="urban", c='g', s=8)
plt.scatter(plot_data[idx_4, 0], plot_data[idx_4, 1], label="open", c='y', s=8)

plt.legend()
plt.axis([-0.2, 3.5, -0.2, 7])
plt.show()