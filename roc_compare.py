# @Time : 2022/3/21 16:38 
# @Author : zhongyu 
# @File : roc_compare.py
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

if __name__ == '__main__':
    roc_data_H_H = np.load('LHdataset\H_L_roc.npy')
    roc_data_L_H = np.load('LHdataset\L_L_5_roc.npy')
    roc_data_LH_H = np.load('LHdataset\LHmix_L_roc.npy')
    roc_data_H10_H = np.load('LHdataset\H10_H_roc.npy')
    # roc_data_rate1 = np.load(r'LHdataset\rate0.5_mix_roc.npy')
    # roc_data_rate2 = np.load(r'LHdataset\LHmix_H_roc.npy')

    # dsp = lgb.Booster(model_file='modeltest/model_L_5_20.txt')
    # print('Plotting feature importances...')
    # ax = lgb.plot_importance(dsp, max_num_features=10)
    # plt.show()

    fig = plt.figure()
    import matplotlib.font_manager as fm

    # 微软雅黑,如果需要宋体,可以用simsun.ttc
    myfont = fm.FontProperties(family='Times New Roman', size=16, weight='bold')
    font = {'family': 'Times New Roman', 'size': 16, 'weight': 'black'}
    plt.plot(roc_data_H_H[0], roc_data_H_H[1], label='H_L', ls='-')
    plt.plot(roc_data_L_H[0], roc_data_L_H[1], label='L_L', ls='--')
    plt.plot(roc_data_LH_H[0], roc_data_LH_H[1], label='LHmix_L', ls='-.')
    # plt.plot(roc_data_H10_H[0], roc_data_H10_H[1], label='H10_H', ls=':')
    # plt.plot(roc_data_rate1[0], roc_data_rate1[1], label='before',marker='o')
    # plt.plot(roc_data_rate2[0], roc_data_rate2[1], label='xue',marker='+')
    # plt.plot(roc_data_rate3[0], roc_data_rate3[1], label='0.6')
    # plt.plot(roc_data_rate4[0], roc_data_rate4[1], label='0.4')
    # plt.plot(roc_data_rate5[0], roc_data_rate5[1], label='0.2')
    # plt.plot(roc_data_rate6[0], roc_data_rate6[1], label='0.5')
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.ylabel('True Positive Rate', fontdict=font)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xticks(fontproperties='Times New Roman', fontsize=12, weight='bold')
    plt.yticks(fontproperties='Times New Roman', fontsize=12, weight='bold')
    plt.legend(loc="lower right", prop=myfont)
