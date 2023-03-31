# @Time : 2022/6/10 15:56 
# @Author : zhongyu 
# @File : jda_application.py
import numpy

from JDA import JDA
import lightgbm as lgb
import pandas as pd
import numpy as np
import tuning_shot as tst
import sklearn
import joblib
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from tsfresh import extract_features, extract_relevant_features, select_features


if __name__ == '__main__':
    df_train_s = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    df_train_t = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    y_train = df_train_s['disrup_tag'].values
    X_train = df_train_s.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_train_t['disrup_tag'].values
    X_val = df_train_t.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train1_data, test_data, train1_y, test_y = \
        train_test_split(X_train, y_train, test_size=0.03, random_state=1, shuffle=True, stratify=y_train)
    train_s_data, test_s_data, train_s_y, test_s_y = \
        train_test_split(X_val, y_val, test_size=0.2, random_state=1, shuffle=True, stratify=y_val)
    jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1,T=10)
    # acc, ypre, list_acc,A_dic = jda.fit_predict(test_data[:,(4,5,6,7)], test_y+1, test_s_data[:,(4,5,6,7)], test_s_y+1)  # 类别需要+1
    acc, ypre, list_acc,A_dic, gbm = jda.lgb_predict(test_data[:,(4,5,6,7)], test_y, test_s_data[:,(4,5,6,7)], test_s_y)  # 类别需要+1
    # np.save(r'LHdataset\JDA_A_maxacc.npy', A_dic[list_acc[1]])

    # model training
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)

    df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    df_val = df_H_val

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    # X_train = tst.trans_a(r'LHdataset\JDA_A_maxacc.npy', X_train)
    # X_val = tst.trans_a(r'LHdataset\JDA_A_maxacc.npy', X_val)
    X_train[:,(4,5,6,7)] = tst.trans_ma(A_dic[list_acc[2]], X_train[:,(4,5,6,7)])
    X_val[:,(4,5,6,7)] = tst.trans_ma(A_dic[list_acc[2]], X_val[:,(4,5,6,7)])
    lgb_train = lgb.Dataset(X_train, y_train.ravel())
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': {'auc'}, 'learning_rate': 0.1,
              'is_unbalance': True, 'min_data_in_leaf': 10,
              'num_leaves': 10,
              'max_depth': 3}

    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets={lgb_train, lgb_eval},
                    evals_result=evals_result,
                    verbose_eval=100,
                    early_stopping_rounds=30)
    Y_tar_pseudo = gbm.predict(X_val)

    threshold = 0.5
    pred_result = []
    for mypred in Y_tar_pseudo:
        if mypred > threshold:
            pred_result.append(1)
        else:
            pred_result.append(0)
    Y_tar_pseudo = np.array(pred_result)
    acc = sklearn.metrics.roc_auc_score(y_val, Y_tar_pseudo)
    print('JDA iteration : Auc: {:.4f}'.format(acc))

