import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import pickle

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
df_indoor_market_gr1['true_class'] = 1
df_indoor_market_gr2['true_class'] = 1
df_indoor_museum_gr1['true_class'] = 1
df_indoor_museum_gr2['true_class'] = 1
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
df_inter_queens_col2 = pd.read_csv(fp_inter_queens_col2_sp)

df_inter_path1['true_class'] = 2
df_inter_path2['true_class'] = 2
df_inter_dept3['true_class'] = 2
df_inter_GreenTS_p1_1['true_class'] = 2
df_inter_GreenTS_p1_2['true_class'] = 2
df_inter_GreenTS_p2_1['true_class'] = 2
df_inter_GreenTS_p2_2['true_class'] = 2
df_inter_market_aw1['true_class'] = 2
df_inter_market_aw2['true_class'] = 2
df_inter_park_dark1['true_class'] = 2
df_inter_park_dark2['true_class'] = 2
df_inter_park_light1['true_class'] = 2
df_inter_park_light2['true_class'] = 2
df_inter_queens_arch['true_class'] = 2
df_inter_queens_col1['true_class'] = 2
df_inter_queens_col2['true_class'] = 2


df_open_park1 = pd.read_csv(fp_open_park1_sp)
df_open_park2 = pd.read_csv(fp_open_park2_sp)

df_open_park1['true_class'] = 4
df_open_park2['true_class'] = 4


df_urban_sl1 = pd.read_csv(fp_urban_sl1_sp)
df_urban_sl2 = pd.read_csv(fp_urban_sl2_sp)
df_urban_cutsark_out1 = pd.read_csv(fp_urban_cutsark_out1_sp)
df_urban_cutsark_out2 = pd.read_csv(fp_urban_cutsark_out2_sp)
df_urban_dept1 = pd.read_csv(fp_urban_dept1_sp)
df_urban_dept2 = pd.read_csv(fp_urban_dept2_sp)
df_urban_GreenTS_p3_1 = pd.read_csv(fp_urban_GreenTS_p3_1_sp)
df_urban_GreenTS_p3_2 = pd.read_csv(fp_urban_GreenTS_p3_2_sp)
df_urban_queens_court = pd.read_csv(fp_urban_queens_court_sp)

df_urban_sl1['true_class'] = 3
df_urban_sl2['true_class'] = 3
df_urban_cutsark_out1['true_class'] = 3
df_urban_cutsark_out2['true_class'] = 3
df_urban_dept1['true_class'] = 3
df_urban_dept2['true_class'] = 3
df_urban_GreenTS_p3_1['true_class'] = 3
df_urban_GreenTS_p3_2['true_class'] = 3
df_urban_queens_court['true_class'] = 3


fp_indoor_bm = '..\data\indoor\British museum\Observations'
fp_indoor_ch2221 = '..\data\indoor\Chadwick 2221\Observations'
fp_indoor_ch103a = '..\data\indoor\Chadwick 103 by the window\A\Observations'
fp_indoor_ch103b = '..\data\indoor\Chadwick 103 by the window\B\Observations'
fp_indoor_jah = '..\data\indoor\JAH363A\Observations'

fp_inter = '..\data\intermediate\P1\Observations'

fp_urban_p1b = '..\data\\urban\P1B\Observations.csv'
fp_urban_p2b = '..\data\\urban\P2B\Observations.csv'
fp_urban_p3b = '..\data\\urban\P3B\Observations.csv'
fp_urban_p4b = '..\data\\urban\P4B\Observations.csv'

fp_open_reg = '..\data\open-sky\P1_REGENT\Observations'
fp_open_hyde = '..\data\open-sky\P4_HYDE\Observations'

df_indoor_bm = pd.read_csv(fp_indoor_bm)
df_indoor_ch2221 = pd.read_csv(fp_indoor_ch2221)
df_indoor_ch103a = pd.read_csv(fp_indoor_ch103a)
df_indoor_ch103b = pd.read_csv(fp_indoor_ch103b)
df_indoor_jah = pd.read_csv(fp_indoor_jah)
df_indoor_bm['true_class'] = 1
df_indoor_ch2221['true_class'] = 1
df_indoor_ch103a['true_class'] = 1
df_indoor_ch103b['true_class'] = 1
df_indoor_jah['true_class'] = 1

df_inter = pd.read_csv(fp_inter)
df_inter['true_class'] = 2

df_urban_p1b = pd.read_csv(fp_urban_p1b)
df_urban_p2b = pd.read_csv(fp_urban_p2b)
df_urban_p3b = pd.read_csv(fp_urban_p3b)
df_urban_p4b = pd.read_csv(fp_urban_p4b)
df_urban_p1b['true_class'] = 3
df_urban_p2b['true_class'] = 3
df_urban_p3b['true_class'] = 3
df_urban_p4b['true_class'] = 3


df_open_reg = pd.read_csv(fp_open_reg)
df_open_hyde = pd.read_csv(fp_open_hyde)
df_open_reg['true_class'] = 4
df_open_hyde['true_class'] = 4

in_samp_1 = df_indoor_bm.sample(100)
in_samp_2 = df_indoor_ch2221.sample(100)
in_samp_3 = df_indoor_ch103a.sample(100)
in_samp_4 = df_indoor_ch103b.sample(100)
in_samp_5 = df_indoor_jah.sample(100)

inter_samp = df_inter.sample(200)

urb_samp1 = df_urban_p1b.sample(100)
urb_samp2 = df_urban_p2b.sample(100)
urb_samp3 = df_urban_p3b.sample(100)
urb_samp4 = df_urban_p4b.sample(100)

#cols = ['obs_id', 'e_id', 'sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
# cols = ['sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25','snr_std', 'elev_0_30', 'elev_30_60', 'elev_60_90',
         'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']

# df = [df_indoor_bm, df_indoor_ch2221, df_indoor_ch103a, df_indoor_ch103b, df_indoor_jah,
#       df_inter, df_urban_p1b, df_urban_p2b, df_urban_p3b, df_urban_p4b, df_open_reg, df_open_hyde]
df = [in_samp_1, in_samp_2, in_samp_3, in_samp_4, in_samp_5, inter_samp, urb_samp1, urb_samp2, urb_samp3, urb_samp4]
data = pd.concat(df).sample(frac=1).reset_index(drop=True)

# with open('orig2_2.pickle', 'wb') as f:
#     pickle.dump(data, f)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
trnsf_data = model.fit_transform(data[cols])

# with open('transf2_3.pickle', 'wb') as f:
#     pickle.dump(trnsf_data, f)
