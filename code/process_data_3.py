import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import csv

# Load the data

# fp_indoor_cutsark_in_1 = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\\raw_T4\Observations.csv'
# fp_indoor_coffee_1 = '..\data_Greenwich\indoor\CoffeeShop\\raw_T4\Observations.csv'
# fp_indoor_station_1 = '..\data_Greenwich\indoor\Greenwich_TrainStation\\raw_T4\Observations.csv'
# fp_indoor_market_gr_1 = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\\raw_T4\Observations.csv'
# fp_indoor_museum_gr_1 = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\\raw_T4\Observations.csv'
# fp_indoor_museum_win_1 = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\\raw_T4\Observations.csv'
#
# fp_inter_path_1 = '..\data_Greenwich\intermediate\covered_path_byGym\\raw_T4\Observations.csv'
# fp_inter_dept3_1 = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\\raw_T4\Observations.csv'
# fp_inter_GreenTS_p1_1 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\\raw_T4\Observations.csv'
# fp_inter_GreenTS_p2_1 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\\raw_T4\Observations.csv'
# fp_inter_park_dark_1 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\\raw_T4\Observations.csv'
# fp_inter_park_light_1 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\\raw_T4\Observations.csv'
# fp_inter_uni_col1_1 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\\raw_T4\Observations.csv'
# fp_inter_uni_col2_1 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\\raw_T4\Observations.csv'
# fp_inter_queens_col_1 = '..\data_Greenwich\intermediate\QueensHouse\colonnade\\raw_T4\Observations.csv'
#
fp_open_park_1 = '..\data_Greenwich\open_sky\GreenwichPark\open\\raw_T4\Observations.csv'
fp_open_park_cric_1 = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\\raw_T4\Observations.csv'
fp_open_blackheath_1 = '..\data_Greenwich\open_sky\Blackheath\\raw_T4\Observations.csv'
fp_open_park_view_1 = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\\raw_T4\Observations.csv'
fp_open_rangers_1 = '..\data_Greenwich\open_sky\RangersHouse\\raw_T4\Observations.csv'
fp_open_hill_1 = '..\data_Greenwich\open_sky\DartmouthHill\\raw_T4\Observations.csv'
#
# fp_urban_sl_1 = '..\data_Greenwich\\urban\\behind_SailLoftPub\\raw_T4\Observations.csv'
# fp_urban_cutsark_out_1 = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\\raw_T4\Observations.csv'
# fp_urban_tesco_1 = '..\data_Greenwich\\urban\ByTesco\\raw_T4\Observations.csv'
# fp_urban_GreenTS_p3_1 = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\\raw_T4\Observations.csv'
# fp_urban_uni_wall_1 = '..\data_Greenwich\\urban\GreenwichUni\\raw_T4\Observations.csv'
# fp_urban_meridian_1 = '..\data_Greenwich\\urban\MeridianPassage\\raw_T4\Observations.csv'
# fp_urban_queens_court_1 = '..\data_Greenwich\\urban\QueensHouse\courtyard\\raw_T4\Observations.csv'
#
# ######
#
# fp_indoor_cutsark_in_2 = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\\raw_T5\Observations.csv'
# fp_indoor_coffee_2 = '..\data_Greenwich\indoor\CoffeeShop\\raw_T5\Observations.csv'
# fp_indoor_station_2 = '..\data_Greenwich\indoor\Greenwich_TrainStation\\raw_T5\Observations.csv'
# fp_indoor_market_gr_2 = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\\raw_T5\Observations.csv'
# fp_indoor_museum_gr_2 = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\\raw_T5\Observations.csv'
# fp_indoor_museum_win_2 = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\\raw_T5\Observations.csv'
#
# fp_inter_path_2 = '..\data_Greenwich\intermediate\covered_path_byGym\\raw_T5\Observations.csv'
# fp_inter_dept3_2 = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\\raw_T5\Observations.csv'
# fp_inter_GreenTS_p1_2 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\\raw_T5\Observations.csv'
# fp_inter_GreenTS_p2_2 = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\\raw_T5\Observations.csv'
# fp_inter_park_dark_2 = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\\raw_T5\Observations.csv'
# fp_inter_uni_col1_2 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\\raw_T5\Observations.csv'
# fp_inter_uni_col2_2 = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\\raw_T5\Observations.csv'
# fp_inter_queens_col_2 = '..\data_Greenwich\intermediate\QueensHouse\colonnade\\raw_T5\Observations.csv'
#
# fp_open_park_2 = '..\data_Greenwich\open_sky\GreenwichPark\open\\raw_T5\Observations.csv'
# fp_open_park_cric_2 = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\\raw_T5\Observations.csv'
# fp_open_blackheath_2 = '..\data_Greenwich\open_sky\Blackheath\\raw_T5\Observations.csv'
fp_open_park_view_2 = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\\raw_T5\Observations.csv'
fp_open_rangers_2 = '..\data_Greenwich\open_sky\RangersHouse\\raw_T5\Observations.csv'
fp_open_hill_2 = '..\data_Greenwich\open_sky\DartmouthHill\\raw_T5\Observations.csv'
#
# fp_urban_sl_2 = '..\data_Greenwich\\urban\\behind_SailLoftPub\\raw_T5\Observations.csv'
# fp_urban_cutsark_out_2 = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\\raw_T5\Observations.csv'
# fp_urban_tesco_2 = '..\data_Greenwich\\urban\ByTesco\\raw_T5\Observations.csv'
# fp_urban_GreenTS_p3_2 = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\\raw_T5\Observations.csv'
# fp_urban_uni_wall_2 = '..\data_Greenwich\\urban\GreenwichUni\\raw_T5\Observations.csv'
# fp_urban_meridian_2 = '..\data_Greenwich\\urban\MeridianPassage\\raw_T5\Observations.csv'
# fp_urban_queens_court_2 = '..\data_Greenwich\\urban\QueensHouse\courtyard\\raw_T5\Observations.csv'
#
# # Save paths
#
fp_indoor_cutsark_in_1_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_ext_4'
fp_indoor_coffee_1_sp = '..\data_Greenwich\indoor\CoffeeShop\Obs_features_ext_4'
fp_indoor_station_1_sp = '..\data_Greenwich\indoor\Greenwich_TrainStation\Obs_features_ext_4'
fp_indoor_market_gr_1_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_ext_4'
fp_indoor_museum_gr_1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_ext_4'
fp_indoor_museum_win_1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\Obs_features_ext_4'

fp_inter_path_1_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_ext_4'
fp_inter_dept3_1_sp = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features_ext_4'
fp_inter_GreenTS_p1_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_ext_4'
fp_inter_GreenTS_p2_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_ext_4'
fp_inter_park_dark_1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_ext_4'
fp_inter_park_light_1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_ext_4'
fp_inter_uni_col1_1_sp = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\Obs_features_ext_4'
fp_inter_uni_col2_1_sp = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\Obs_features_ext_4'
fp_inter_queens_col_1_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_ext_4'

fp_open_park_1_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_ext_4'
fp_open_park_cric_1_sp = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\Obs_features_ext_4'
fp_open_blackheath_1_sp = '..\data_Greenwich\open_sky\Blackheath\Obs_features_ext_4'
fp_open_park_view_1_sp = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\Obs_features_ext_4'
fp_open_rangers_1_sp = '..\data_Greenwich\open_sky\RangersHouse\Obs_features_ext_4'
fp_open_hill_1_sp = '..\data_Greenwich\open_sky\DartmouthHill\Obs_features_ext_4'

fp_urban_sl_1_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_ext_4'
fp_urban_cutsark_out_1_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_ext_4'
fp_urban_tesco_1_sp = '..\data_Greenwich\\urban\ByTesco\Obs_features_ext_4'
fp_urban_GreenTS_p3_1_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_ext_4'
fp_urban_uni_wall_1_sp = '..\data_Greenwich\\urban\GreenwichUni\Obs_features_ext_4'
fp_urban_meridian_1_sp = '..\data_Greenwich\\urban\MeridianPassage\Obs_features_ext_4'
fp_urban_queens_court_1_sp = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features_ext_4'

######

fp_indoor_cutsark_in_2_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_ext_5'
fp_indoor_coffee_2_sp = '..\data_Greenwich\indoor\CoffeeShop\Obs_features_ext_5'
fp_indoor_station_2_sp = '..\data_Greenwich\indoor\Greenwich_TrainStation\Obs_features_ext_5'
fp_indoor_market_gr_2_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_ext_5'
fp_indoor_museum_gr_2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_ext_5'
fp_indoor_museum_win_2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\by_window\Obs_features_ext_5'

fp_inter_path_2_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_ext_5'
fp_inter_dept3_2_sp = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features_ext_5'
fp_inter_GreenTS_p1_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_ext_5'
fp_inter_GreenTS_p2_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_ext_5'
fp_inter_park_dark_2_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_ext_5'
fp_inter_uni_col1_2_sp = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeInsideCorner\Obs_features_ext_5'
fp_inter_uni_col2_2_sp = '..\data_Greenwich\intermediate\GreenwichUniversity\ColonnadeOutsideCorner\Obs_features_ext_5'
fp_inter_queens_col_2_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_ext_5'

fp_open_park_2_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_ext_5'
fp_open_park_cric_2_sp = '..\data_Greenwich\open_sky\GreenwichPark\cricket_field\Obs_features_ext_5'
fp_open_blackheath_2_sp = '..\data_Greenwich\open_sky\Blackheath\Obs_features_ext_5'
fp_open_park_view_2_sp = '..\data_Greenwich\open_sky\GreenwichPark\\view_point\Obs_features_ext_5'
fp_open_rangers_2_sp = '..\data_Greenwich\open_sky\RangersHouse\Obs_features_ext_5'
fp_open_hill_2_sp = '..\data_Greenwich\open_sky\DartmouthHill\Obs_features_ext_5'

fp_urban_sl_2_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_ext_5'
fp_urban_cutsark_out_2_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_ext_5'
fp_urban_tesco_2_sp = '..\data_Greenwich\\urban\ByTesco\Obs_features_ext_5'
fp_urban_GreenTS_p3_2_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_ext_5'
fp_urban_uni_wall_2_sp = '..\data_Greenwich\\urban\GreenwichUni\Obs_features_ext_5'
fp_urban_meridian_2_sp = '..\data_Greenwich\\urban\MeridianPassage\Obs_features_ext_5'
fp_urban_queens_court_2_sp = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features_ext_5'


def create_feature_file(data_path, save_path):
    df_data = pd.read_csv(data_path)

    # Clean data by deleting entries with 0 values
    df_clean = df_data.query('(CN0 != 0.0) & (elevation != 0.0)')

    # Generate features
    grouped = df_clean.groupby('e_id')
    feature_df = pd.DataFrame(index=grouped.indices, columns=['e_id', 'num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25',
                                                              'num_sat_u20', 'sum_sat_u20', 'snr_std',
                                                              'elev_0_30', 'elev_30_60', 'elev_60_90',
                                                              'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
                                                              'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
                                                              'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25'])

    for epoch_id, group in grouped:
        over_thresh = group[group.CN0 >= 25.0]
        under_thresh = group[group.CN0 <= 20.0]
        feature_df.loc[epoch_id] = [epoch_id, group.shape[0], group.CN0.sum(), over_thresh.shape[0], over_thresh.CN0.sum(),
                                    under_thresh.shape[0], under_thresh.CN0.sum(), group.CN0.std(),
                                    group[group.elevation <= 30.0].shape[0],
                                    group[(group.elevation <= 60.0) & (group.elevation > 30.0)].shape[0],
                                    group[group.elevation > 60.0].shape[0],
                                    group[group.elevation <= 30.0].CN0.sum(),
                                    group[(group.elevation <= 60.0) & (group.elevation > 30.0)].CN0.sum(),
                                    group[group.elevation > 60.0].CN0.sum(),
                                    over_thresh[over_thresh.elevation <= 30.0].shape[0],
                                    over_thresh[(over_thresh.elevation <= 60.0) & (over_thresh.elevation > 30.0)].shape[0],
                                    over_thresh[over_thresh.elevation > 60.0].shape[0],
                                    over_thresh[over_thresh.elevation <= 30.0].CN0.sum(),
                                    over_thresh[(over_thresh.elevation <= 60.0) & (over_thresh.elevation > 30.0)].CN0.sum(),
                                    over_thresh[over_thresh.elevation > 60.0].CN0.sum()
        ]
    feature_df.to_csv(save_path)


# Create feature files
#
# create_feature_file(fp_indoor_cutsark_in_1, fp_indoor_cutsark_in_1_sp)
# create_feature_file(fp_indoor_coffee_1, fp_indoor_coffee_1_sp)
# create_feature_file(fp_indoor_station_1, fp_indoor_station_1_sp)
# create_feature_file(fp_indoor_market_gr_1, fp_indoor_market_gr_1_sp)
# create_feature_file(fp_indoor_museum_gr_1, fp_indoor_museum_gr_1_sp)
# create_feature_file(fp_indoor_museum_win_1, fp_indoor_museum_win_1_sp)
#
# create_feature_file(fp_inter_path_1, fp_inter_path_1_sp)
# create_feature_file(fp_inter_dept3_1, fp_inter_dept3_1_sp)
# create_feature_file(fp_inter_GreenTS_p1_1, fp_inter_GreenTS_p1_1_sp)
# create_feature_file(fp_inter_GreenTS_p2_1, fp_inter_GreenTS_p2_1_sp)
# create_feature_file(fp_inter_park_dark_1, fp_inter_park_dark_1_sp)
# create_feature_file(fp_inter_park_light_1, fp_inter_park_light_1_sp)
# create_feature_file(fp_inter_uni_col1_1, fp_inter_uni_col1_1_sp)
# create_feature_file(fp_inter_uni_col2_1, fp_inter_uni_col2_1_sp)
# create_feature_file(fp_inter_queens_col_1, fp_inter_queens_col_1_sp)
#
# create_feature_file(fp_open_park_1, fp_open_park_1_sp)
# create_feature_file(fp_open_park_cric_1, fp_open_park_cric_1_sp)
# create_feature_file(fp_open_blackheath_1, fp_open_blackheath_1_sp)
create_feature_file(fp_open_park_view_1, fp_open_park_view_1_sp)
create_feature_file(fp_open_rangers_1, fp_open_rangers_1_sp)
# create_feature_file(fp_open_hill_1, fp_open_hill_1_sp)
#
# create_feature_file(fp_urban_sl_1, fp_urban_sl_1_sp)
# create_feature_file(fp_urban_cutsark_out_1, fp_urban_cutsark_out_1_sp)
# create_feature_file(fp_urban_tesco_1, fp_urban_tesco_1_sp)
# create_feature_file(fp_urban_GreenTS_p3_1, fp_urban_GreenTS_p3_1_sp)
# create_feature_file(fp_urban_uni_wall_1, fp_urban_uni_wall_1_sp)
# create_feature_file(fp_urban_meridian_1, fp_urban_meridian_1_sp)
# create_feature_file(fp_urban_queens_court_1, fp_urban_queens_court_1_sp)

########
#
# create_feature_file(fp_indoor_cutsark_in_2, fp_indoor_cutsark_in_2_sp)
# create_feature_file(fp_indoor_coffee_2, fp_indoor_coffee_2_sp)
# create_feature_file(fp_indoor_station_2, fp_indoor_station_2_sp)
# create_feature_file(fp_indoor_market_gr_2, fp_indoor_market_gr_2_sp)
# create_feature_file(fp_indoor_museum_gr_2, fp_indoor_museum_gr_2_sp)
# create_feature_file(fp_indoor_museum_win_2, fp_indoor_museum_win_2_sp)
#
# create_feature_file(fp_inter_path_2, fp_inter_path_2_sp)
# create_feature_file(fp_inter_dept3_2, fp_inter_dept3_2_sp)
# create_feature_file(fp_inter_GreenTS_p1_2, fp_inter_GreenTS_p1_2_sp)
# create_feature_file(fp_inter_GreenTS_p2_2, fp_inter_GreenTS_p2_2_sp)
# create_feature_file(fp_inter_park_dark_2, fp_inter_park_dark_2_sp)
# create_feature_file(fp_inter_uni_col1_2, fp_inter_uni_col1_2_sp)
# create_feature_file(fp_inter_uni_col2_2, fp_inter_uni_col2_2_sp)
# create_feature_file(fp_inter_queens_col_2, fp_inter_queens_col_2_sp)
#
# create_feature_file(fp_open_park_2, fp_open_park_2_sp)
# create_feature_file(fp_open_park_cric_2, fp_open_park_cric_2_sp)
# create_feature_file(fp_open_blackheath_2, fp_open_blackheath_2_sp)
# create_feature_file(fp_open_park_view_2, fp_open_park_view_2_sp)
# create_feature_file(fp_open_rangers_2, fp_open_rangers_2_sp)
# create_feature_file(fp_open_hill_2, fp_open_hill_2_sp)
#
# create_feature_file(fp_urban_sl_2, fp_urban_sl_2_sp)
# create_feature_file(fp_urban_cutsark_out_2, fp_urban_cutsark_out_2_sp)
# create_feature_file(fp_urban_tesco_2, fp_urban_tesco_2_sp)
# create_feature_file(fp_urban_GreenTS_p3_2, fp_urban_GreenTS_p3_2_sp)
# create_feature_file(fp_urban_uni_wall_2, fp_urban_uni_wall_2_sp)
# create_feature_file(fp_urban_meridian_2, fp_urban_meridian_2_sp)
# create_feature_file(fp_urban_queens_court_2, fp_urban_queens_court_2_sp)