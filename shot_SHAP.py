# @Time : 2022/3/29 15:30 
# @Author : zhongyu 
# @File : shot_SHAP.py
import lightgbm as lgb
import shap
import test_analysis_shot as tas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import hdf5Reader2A as h5r
import file_read as fr

if __name__ == '__main__':
    print('Loading data&model...')
    # load or create your dataset
    H_beta_shot = np.load(r'LHdataset\H_beta_test.npy')
    df_validation = pd.read_csv('LHdataset/topdata_H_test.csv', index_col=0)

    validset_b = H_beta_shot

    dsp = lgb.Booster(model_file='model/model_H.txt')
    dsp.params["objective"] = "binary"
    explainer = shap.TreeExplainer(dsp)
    features = ['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"]

    # # 画出每一炮的ip与预测值
    # for i in range(validset_b.shape[0]):
    #     delta_t = 3
    #     a1 = 0.9
    #     va_shot = validset_b[i, 0]
    #     # time, data_ip = h5r.read_channel(va_shot, channel='IP', device="2a")
    #     time, data_ip = fr.read_data(va_shot, "IP", 0, 2.5)
    #     if validset_b[i, 1]:
    #         validset_b[i, 1] = 1
    #     dis = df_validation[df_validation['#'] == va_shot]
    #     X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    #     y = dsp.predict(X, num_iteration=dsp.best_iteration)
    #     time_shot = dis['time'].values
    #     shot_predict = 0
    #     for j in range(len(y) - delta_t):
    #         subset = y[j:j + delta_t]
    #         if subset.min() > a1:
    #             shot_predict = 1
    #             break
    #     if shot_predict:
    #         t_warn = time_shot[j + delta_t]
    #     else:
    #         t_warn = 0
    #
    #     fig = plt.figure()
    #     import matplotlib.font_manager as fm
    #
    #     # 微软雅黑,如果需要宋体,可以用simsun.ttc
    #     myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
    #     fronts = {'family': 'Times New Roman', 'size': 12}
    #     ax1 = fig.add_subplot(111)
    #     plt.title('#{}'.format(va_shot))
    #     plt.plot(time, data_ip, label='Ip')
    #     plt.xlabel('time', fontproperties=myfont)
    #     plt.ylabel('Ip', fontproperties=myfont)
    #     plt.xticks(fontproperties='Times New Roman', fontsize=12)
    #     plt.yticks(fontproperties='Times New Roman', fontsize=12)
    #     ax2 = ax1.twinx()
    #     ax2.plot(time_shot, y, label='prediction', color='k')
    #     plt.axvline(x=t_warn, color='r')
    #     plt.ylim([0.0, 1.0])
    #     plt.savefig('shap_fig/{}.png'.format(va_shot))
    #     plt.close(fig)

    shot_for_shap = [36755, 36823, 37021]
    # shot_for_shap = [36755]
    feature_opposite = ["BOLD03", "BOLD06", "BOLU03", "BOLU06", "EFIT_BETA_T"]
    feature_similar = ["DENSITY", "W_E", "V_LOOP", "FIR01", "FIR03"]
    feature_dict = {"BOLD03": 4, "BOLD06": 5, "BOLU03": 6, "BOLU06": 7, "EFIT_BETA_T": 10, "DENSITY": 17, "W_E": 18,
                    "V_LOOP": 3, "FIR01": 19, "FIR03": 20}
    for k in shot_for_shap:
        va_shot = k
        # time, data_ip = h5r.read_channel(va_shot, channel='IP', device="2a")
        time, data_ip = fr.read_data(va_shot, "IP", 0, 2.5)
        dis = df_validation[df_validation['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y = dsp.predict(X, num_iteration=dsp.best_iteration)
        time_shot = dis['time'].values

        shap_values = explainer.shap_values(X)
        # plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X, show=False, feature_names=features)
        # shap.save_html("high_beta_fig\index_{}.htm".format(va_shot), plot)  # force_plot保存
        fig = plt.figure()
        import matplotlib.font_manager as fm

        # 微软雅黑,如果需要宋体,可以用simsun.ttc
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        fronts = {'family': 'Times New Roman', 'size': 12}
        ax1 = fig.add_subplot(311)
        plt.title('#{}'.format(va_shot))
        plt.plot(time, data_ip, label='Ip')
        plt.xlabel('time', fontproperties=myfont)
        plt.ylabel('Ip', fontproperties=myfont)
        plt.xticks(fontproperties='Times New Roman', fontsize=12)
        plt.yticks(fontproperties='Times New Roman', fontsize=12)
        ax2 = ax1.twinx()
        ax2.plot(time_shot, y, label='prediction', color='k')
        plt.yticks(fontproperties='Times New Roman', fontsize=12)
        plt.ylim([0.0, 1.0])
        ax3 = fig.add_subplot(312)
        plt.title('#{} opposite'.format(va_shot))
        for feature_plot in feature_opposite:
            plt.plot(time_shot, shap_values[1][:, feature_dict[feature_plot]], label=feature_plot)
        plt.xlabel('time', fontproperties=myfont)
        plt.ylabel('SHAP value', fontproperties=myfont)
        plt.xticks(fontproperties='Times New Roman', fontsize=12)
        plt.yticks(fontproperties='Times New Roman', fontsize=12)
        plt.legend()
        ax4 = fig.add_subplot(313)
        plt.title('#{} similar'.format(va_shot))
        for feature_plot in feature_similar:
            plt.plot(time_shot, shap_values[1][:, feature_dict[feature_plot]], label=feature_plot)
        plt.xlabel('time', fontproperties=myfont)
        plt.ylabel('SHAP value', fontproperties=myfont)
        plt.xticks(fontproperties='Times New Roman', fontsize=12)
        plt.yticks(fontproperties='Times New Roman', fontsize=12)
        plt.legend()

    # # a1 取0.93 for model_H
    # level = np.linspace(0, 1, 50)
    # level = np.sort(np.append(level, [0.98, 0.981, 0.982, 0.995, 0.996, 0.997, 0.998, 0.999]))
    # max_f1 = float('0')
    # best_params = {}
    # Fpr = []
    # Tpr = []
    # for a1 in level:
    #     predict_result = tas.assess1(validset_b, df_validation, a1, 1, dsp)
    #     f1 = metrics.f1_score(validset_b[:, 1], predict_result[:, 1])
    #     if f1 >= max_f1:
    #         max_f1 = f1
    #         best_params['a1'] = a1
