# @Time : 2022/2/19 14:31 
# @Author : zhongyu 
# @File : params_tuning.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import bayesHyperTuning as bht


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', fbeta_score(y_true, y_hat, beta=1), True


def train_assess(params, lgb_train, lgb_eval, validset_b, df_validation, a1, delta_t):
    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=[lgb_train, lgb_eval],
                    evals_result=evals_result,
                    verbose_eval=100,
                    early_stopping_rounds=30)
    lgb.plot_metric(booster=evals_result, metric='auc')
    gbm.save_model('model/model_L_test_consistence.txt')
    predict_result = tas.assess1(validset_b, df_validation, a1, delta_t, gbm)
    return predict_result


if __name__ == '__main__':
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    df_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    testset_b = np.load(r'dataset\train_dis_shot.npy')
    testset_a = np.load(r'dataset\train_undis_shot.npy')
    # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    validset_b = np.load(r'LHdataset\L_beta_val.npy')

    # # high_beta
    # df_train = pd.read_csv('LHdataset/topdata_H_train.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # # testset_b = np.load(r'dataset\train_dis_shot.npy')
    # # testset_a = np.load(r'dataset\train_undis_shot.npy')
    # validset_b = np.load(r'LHdataset\H_beta_val.npy')

    # # 混合L和H数据集
    # df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    # df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # weight_1 = np.ones([df_train.shape[0]])
    # weight_2 = np.ones([df_H_beta_mix.shape[0]]) * 1.5
    # w_train = list(np.append(weight_1, weight_2, axis=0))
    # df_L_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    # df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\L_beta_val.npy')
    # validset_a = np.load(r'LHdataset\H_beta_val.npy')
    # # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    # validset_b = np.append(validset_b, validset_a, axis=0)
    # newdf_H = pd.DataFrame(np.repeat(df_H_beta_mix.values, 6, axis=0))
    # newdf_H.columns = df_H_beta_mix.columns
    # df_train = df_train.append(newdf_H, ignore_index=True)
    # # df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    # df_val = df_L_val.append(df_H_val, ignore_index=True)

    # # 10炮训练集
    # df_train = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\H_beta_val.npy')

    # save_tag = input("save it or not?")

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train,
                            feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                          "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                          "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                          "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # # bayes调超参
    # bht.optimize_lgb(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 5,
        'num_leaves': 20,
        'learning_rate': 0.1,
        'feature_fraction': 0.86,
        'bagging_fraction': 0.73,
        'min_data_in_leaf': 165,
        'lambda_l1': 0.03,
        'lambda_l2': 2.78,'boost_from_average': False,
        'is_unbalance': True
    }

    a1 = 0.5
    delta_t = 90
    predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')

    # evals_result = {}  # to record eval results for plotting
    #
    # print('Starting training...')
    # # train
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=100,
    #                 valid_sets=lgb_eval,
    #                 evals_result=evals_result,
    #                 early_stopping_rounds=20)

    # # 自定义metrics
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=300,
    #                 valid_sets=[lgb_train, lgb_eval],
    #                 feval=lgb_f1_score,
    #                 evals_result=evals_result,
    #                 verbose_eval=100,
    #                 early_stopping_rounds=30)
