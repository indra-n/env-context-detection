import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_data = pickle.load(f)


indoor = train_data[train_data['true_class'] == 1]
inter = train_data[train_data['true_class'] == 2]
urban = train_data[train_data['true_class'] == 3]
open_sky = train_data[train_data['true_class'] == 4]

indoor_norm_counts = (indoor['num_sat'].value_counts()/indoor.shape[0])*100

fig, ((plt_indoor, plt_inter), (plt_urban, plt_open)) = plt.subplots(2, 2)
fig.tight_layout(pad=1.0, w_pad=4.0, h_pad=1.0)

indoor_norm_30_counts = (indoor['elev_60_90'].value_counts()/indoor.shape[0])*100
#plt_indoor.figure()
plt_indoor.bar(indoor_norm_30_counts.index.tolist(), indoor_norm_30_counts)
plt_indoor.set_ylabel('Normalized counts (%)', fontsize=8)
plt_indoor.set_xlabel('Number of satellites', fontsize=8)
plt_indoor.axis([-0.5, 15, 0, 28])
plt_indoor.minorticks_on()
plt_indoor.grid(True, which='both', linestyle=':', linewidth='0.5')
plt_indoor.tick_params(which='both', top='off', left='off', right='off', bottom='off')
plt_indoor.set_axisbelow(True)
plt_indoor.set_xticks([0, 3, 6, 9, 12, 15])

inter_norm_30_counts = (inter['elev_60_90'].value_counts()/inter.shape[0])*100
#plt_inter.figure()
plt_inter.set_title("Intermediate environment elev under 30", fontsize=10)
plt_inter.bar(inter_norm_30_counts.index.tolist(), inter_norm_30_counts)
plt_inter.set_ylabel('Normalized counts (%)', fontsize=8)
plt_inter.set_xlabel('Number of satellites', fontsize=8)
plt_inter.axis([-0.5, 15, 0, 28])
plt_inter.minorticks_on()
plt_inter.grid(True, which='both', linestyle=':', linewidth='0.5')
plt_inter.tick_params(which='both', top='off', left='off', right='off', bottom='off')
plt_inter.set_axisbelow(True)
plt_inter.set_xticks([0, 3, 6, 9, 12, 15])

urban_norm_30_counts = (urban['elev_60_90'].value_counts()/urban.shape[0])*100
#plt_urban.figure()
plt_urban.set_title("Urban environment elev under 30", fontsize=10)
plt_urban.bar(urban_norm_30_counts.index.tolist(), urban_norm_30_counts)
plt_urban.set_ylabel('Normalized counts (%)', fontsize=8)
plt_urban.set_xlabel('Number of satellites', fontsize=8)
plt_urban.axis([-0.5, 15, 0, 28])
plt_urban.minorticks_on()
plt_urban.grid(True, which='both', linestyle=':', linewidth='0.5')
plt_urban.tick_params(which='both', top='off', left='off', right='off', bottom='off')
plt_urban.set_axisbelow(True)
plt_urban.set_xticks([0, 3, 6, 9, 12, 15])


open_sky_norm_30_counts = (open_sky['elev_60_90'].value_counts()/open_sky.shape[0])*100
#plt_open.figure()
plt_open.set_title("Open environment elev under 30", fontsize=10)
plt_open.bar(open_sky_norm_30_counts.index.tolist(), open_sky_norm_30_counts)
plt_open.set_ylabel('Normalized counts (%)', fontsize=8)
plt_open.set_xlabel('Number of satellites', fontsize=8)
plt_open.axis([-0.5, 15, 0, 28])
plt_open.minorticks_on()
plt_open.grid(True, which='both', linestyle=':', linewidth='0.5')
plt_open.tick_params(which='both', top='off', left='off', right='off', bottom='off')
plt_open.set_axisbelow(True)
plt_open.set_xticks([0, 3, 6, 9, 12, 15])

plt.show()


