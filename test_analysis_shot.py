# @Time : 2022/2/18 17:03 
# @Author : zhongyu 
# @File : test_analysis_shot.py


import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import inference
import joblib
import tuning_shot as tst
import sklearn
import coral_application as capp
import data_process as dap


def assess1(validset_b, df_validation, a1, delta_t, dsp):
    predict_result = np.empty([0, 3])

    for i in range(validset_b.shape[0]):
        va_shot = validset_b[i, 0]
        if validset_b[i, 1]:
            validset_b[i, 1] = 1
        dis = df_validation[df_validation['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y = dsp.predict(X, num_iteration=dsp.best_iteration)
        time_shot = dis['time'].values
        shot_predict = 0
        for j in range(len(y) - delta_t):
            subset = y[j:j + delta_t]
            if subset.min() > a1:
                shot_predict = 1
                break
        if shot_predict:
            t_warn = time_shot[j + delta_t]
        else:
            t_warn = 0
        predict_result = np.append(predict_result, [[va_shot, shot_predict, t_warn]], axis=0)
    return predict_result


def xue_assess(validset_b, df_validation, dsp):
    predict_result = dict()

    for i in range(validset_b.shape[0]):
        va_shot = validset_b[i, 0]
        if validset_b[i, 1]:
            validset_b[i, 1] = 1
        dis = df_validation[df_validation['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

        y = dsp.predict(X, num_iteration=dsp.best_iteration)
        predict_result.setdefault(va_shot, []).append(y)
        predict_result.setdefault(va_shot, []).append(validset_b[i, 1])
    return predict_result


def xue_assess_jda(validset_b, df_validation, dsp):
    predict_result = dict()
    for i in range(validset_b.shape[0]):
        va_shot = validset_b[i, 0]
        if validset_b[i, 1]:
            validset_b[i, 1] = 1
        dis = df_validation[df_validation['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        X = tst.trans_a(r'LHdataset\JDA_A_maxacc.npy', X)
        y = dsp.predict(X)
        predict_result.setdefault(va_shot, []).append(y)
        predict_result.setdefault(va_shot, []).append(validset_b[i, 1])
    return predict_result


if __name__ == '__main__':
    print('Loading data&model...')
    # load or create your dataset
    H_beta_shot = np.load(r'LHdataset\t2_L_beta_test.npy')
    error_d_shot = np.load(r'LHdataset\t2_L&H_win_errorshot.npy')
    H_beta_shot = dap.error_del(H_beta_shot, error_d_shot)
    # df_validation = pd.read_csv('LHdataset/topdata_H_test.csv', index_col=0)
    df_validation = pd.read_csv('LHdataset/t2_win_topdata_test.csv', index_col=0)
    # df_validation_add = pd.read_csv('LHdataset/t2_win_topdata_test_add.csv', index_col=0)
    # df_validation = df_validation.join(df_validation_add)
    roc_data2a7 = np.load('LHdataset\L_H_mix_roc.npy')
    validset_b = H_beta_shot

    # 模型加载
    dsp = lgb.Booster(model_file='model/model_t2_win_L.txt')

    # # 1.预警时间
    # a1 = 0.9
    # delta_t = 1
    # predict_result = assess1(validset_b, df_validation, a1, delta_t, dsp)
    # prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_result[:, 1], average='binary')
    # print(metrics.classification_report(validset_b[:, 1], predict_result[:, 1]))
    # import seaborn as sns
    # warn_time = []
    # a=0
    # b=0
    # c=0
    # for i in range(validset_b.shape[0]):
    #     if predict_result[i,1] and validset_b[i,1]:
    #         time_w = validset_b[i,2]/1000-predict_result[i,2]
    #         warn_time.append(time_w)
    #         if time_w>0.005 and time_w<=0.1:
    #             a+=1
    #         if time_w>0.1 and time_w<=0.3:
    #             b+=1
    #         if time_w>0.3:
    #             c+=1
    # ax = sns.distplot(warn_time)

    # # 2.Tp&Fp best:a1=.7,delta_t=6,f1=0.76   2022/2/23
    # Fpr=[]
    # Tpr=[]
    # max_auc = float('0')
    # best_params ={}
    # # shot_predict, precision, recall, f1,fpr,tpr,pr_time = judge_figure(df_testvt,shotnum,test_result ,0.897, 10,0.4)
    # for a1 in [0.5,0.6,0.7,0.8,0.9]:
    #     for delta_t in [1,2,3,4,5,6]:
    #         predict_result = assess1(validset_b, df_validation, a1, delta_t, dsp)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_result[:, 1], average='binary')
    #         if prf_r[2] >= max_auc:
    #             max_auc = prf_r[2]
    #             best_params['a1'] = a1
    #             best_params['delta_t'] = delta_t
    #         tn, fp, fn, tp = metrics.confusion_matrix(validset_b[:, 1],  predict_result[:, 1]).ravel()
    #         tpr = tp / (tp + fn)
    #         fpr = fp / (fp + tn)
    #         Fpr.append(fpr)
    #         Tpr.append(tpr)
    # TPFP = np.array([Tpr, Fpr])
    # TPFP = TPFP[np.argsort(TPFP[:,0])]

    # 3.roc
    level = np.linspace(0, 1, 50)
    level = np.sort(np.append(level, [0.98, 0.983, 0.987, 0.99, 0.995]))
    max_auc = float('0')
    best_params = {}
    Fpr = []
    Tpr = []

    predict_result = xue_assess(validset_b, df_validation, dsp)
    for a1 in level:
        # predict_result = assess1(validset_b, df_validation, a1, 1, dsp)
        # tn, fp, fn, tp = metrics.confusion_matrix(validset_b[:, 1], predict_result[:, 1]).ravel()
        # tpr = tp / (tp + fn)
        # fpr = fp / (fp + tn)
        # Fpr.append(fpr)
        # Tpr.append(tpr)

        tpr, fpr, pre_time = inference.evaluation(predict_result, a1)
        Fpr.append(fpr)
        Tpr.append(tpr)
    roc_data = [Fpr, Tpr]

    # np.save(r'LHdataset\rate0.2_mix_roc.npy', roc_data)

    fig = plt.figure()
    import matplotlib.font_manager as fm

    # 微软雅黑,如果需要宋体,可以用simsun.ttc
    myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
    fronts = {'family': 'Times New Roman', 'size': 12}
    plt.plot(Fpr, Tpr, label='LmixH_H')
    # plt.plot(roc_data2a7[:,1], roc_data2a7[:,0], label='L_H', linestyle='-.')
    plt.xlabel('Fpr', fontproperties=myfont)
    plt.ylabel('Tpr', fontproperties=myfont)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xticks(fontproperties='Times New Roman', fontsize=12)
    plt.yticks(fontproperties='Times New Roman', fontsize=12)
