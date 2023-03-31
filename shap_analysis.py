# @Time : 2022/2/23 19:08 
# @Author : zhongyu 
# @File : shap_analysis.py
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def shap_summary(data_file, model_f):
    """
    :param data_file:
    :param model_f:
    :return:
    """
    df_validation = pd.read_csv(data_file, index_col=0)
    dsp = lgb.Booster(model_file=model_f)

    test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
    feature_name = ['${ΔI_p}$', '${β_N}$', "${H_α}$", "${V_LOOP}$", "BOLD03", "BOLD06", "BOLU03",
                    "BOLU06", "SXR03", "SXR06", "EFIT_${β_T}$", "EFIT_${β_P}$",
                    "EFIT_ELONGATION", "EFIT_${L_i}$", "EFIT_${q_0}$", "EFIT_${q_BDRY}$",
                    "${B_t}$", "DENSITY04", "${W_E}$", "DENSITY01", "DENSITY03"]
    dsp.params["objective"] = "binary"
    explainer = shap.TreeExplainer(dsp)
    shap_values = explainer.shap_values(test_data)

    plt.figure()
    shap.summary_plot(shap_values[1], test_data, feature_names=feature_name, max_display=15)
    plt.tight_layout()


if __name__ == '__main__':
    # print('Loading data&model...')
    # # load or create your dataset
    # # # shap_summary各模型对比
    # data_L = 'LHdataset/topdata_test.csv'
    # data_H = 'LHdataset/topdata_H_test.csv'
    #
    # model_L = 'model/model_L_5_20.txt'
    # model_H = 'model/model_H.txt'
    # model_mix = 'model/modelL_H_mix9_100.txt'
    # model_10 = 'model/model_H10.txt'
    # # shap_summary(data_L, model_L)
    # # shap_summary(data_H, model_H)
    # # shap_summary(data_H, model_mix)
    # shap_summary(data_H, model_mix)

    # shap分析
    df_validation = pd.read_csv('LHdataset/topdata_H_test.csv', index_col=0)

    # dsp = lgb.Booster(model_file='model/model_1.txt')
    # dsp = lgb.Booster(model_file='model/model_1_10_40_300.txt')
    dsp = lgb.Booster(model_file='model/model_H.txt')

    test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
    y = df_validation['disrup_tag']
    dsp.params["objective"] = "binary"
    explainer = shap.TreeExplainer(dsp)
    shap_values = explainer.shap_values(test_data)

    import matplotlib.font_manager as fm

    # 微软雅黑,如果需要宋体,可以用simsun.ttc
    myfont = fm.FontProperties(family='Times New Roman', size=20, weight='bold')
    font = {'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}

    # # 特征聚类
    # clustering = shap.utils.hclust(test_data, y)
    # shap.plots.bar(shap_values,
    #                clustering=clustering,
    #                clustering_cutoff=0.5)

    # plt.figure(dpi=1200)
    # fig = plt.gcf()
    # shap.summary_plot(shap_values[1], test_data, plot_type="bar", show=False)
    # plt.tight_layout()
    # plt.savefig('filename.png')
    # fig = plt.gcf()
    shap.dependence_plot("BOLU06", shap_values[1], test_data,interaction_index=None,show=False)
    plt.title("H_H",fontdict=font)
    plt.ylabel("SHAP value for the 'BOLU06'",fontdict=font)
    plt.xlabel('BOLU06(a.u.)',fontdict=font)
    plt.xticks(fontproperties='Times New Roman', fontsize=20, weight='bold')
    plt.yticks(fontproperties='Times New Roman', fontsize=20, weight='bold')
    # plt.savefig("my_dependence_plot.pdf") # we can save a PDF of the figure if we want
    plt.show()
    # plt.tight_layout()

    # for name in columns:
    #     plt.figure(dpi=1200)
    #     fig = plt.gcf()
    #     shap.dependence_plot(name, shap_values[1], test_data, interaction_index="BT", show=False)
    #     plt.tight_layout()
    #     plt.savefig('shap_fig/{}.png'.format(name))
    #     plt.close(fig)
