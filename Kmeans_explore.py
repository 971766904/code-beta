# @Time : 2022/5/31 10:11 
# @Author : zhongyu 
# @File : Kmeans_explore.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from file_read import read_data


def kmeanstool(shot, dataset):
    shotdata = dataset.loc[dataset['#'].isin(shot)]
    y_train = shotdata['disrup_tag']
    time = shotdata['time']
    X_train = shotdata.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
    pred = kmeans.labels_
    plt.figure()
    plt.plot(time, pred)


if __name__ == '__main__':
    df_H = pd.read_csv('LHdataset/topdata_H_beta.csv', index_col=0)
    validset_b = np.load(r'LHdataset\H_beta.npy')
    disrupt_train = [36387, 36389, 36391, 36761, 36765, 36766, 36767, 36785, 36790, 36795, 36797, 36800, 36801, 36802,
                     36803, 36804, 36805, 36806, 36825, 36826, 36827, 36828, 36830, 36837, 36838, 36840, 36846, 36847,
                     36848, 36849, 36850, 36865, 36866, 36870, 36871, 36873, 36888, 36890, 36891, 36892, 36893, 36894]
    disrupt_shot = [36897, 36898, 36899, 36939, 36941, 36945, 36946, 36947, 36964, 36965, 36966, 36967,
                    36968, 36969, 36970, 36972, 36981, 36983, 36992, 37003, 37004, 37005, 37007, 37013, 37014, 37020,
                    37021, 37022, 37023, 37030, 37031, 37080]

    for shot in disrupt_train:
        mix_shot = df_H.loc[df_H['#'].isin([shot])]
        time = mix_shot['time'].values
        X_train = mix_shot.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
        pred = kmeans.labels_
        time_ip, data = read_data(shot, "IP", time[0], time[-1] + 0.2)
        plt.figure()
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        plt.plot(time_ip, data, 'r', label='Ip')
        plt.ylabel('Ip', fontproperties=myfont)
        # axes1.set_yticks(fontproperties='Times New Roman', fontsize=12)
        axes2 = plt.twinx()
        axes2.plot(time, pred, 'b', label='label')
        axes2.set_ylabel('k_label', fontproperties=myfont)
        plt.savefig('kmeans/{}.png'.format(shot), bbox_inches='tight')
        plt.close()
