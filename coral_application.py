# @Time : 2022/6/24 16:22 
# @Author : zhongyu 
# @File : coral_application.py
from CORAL import CORAL
import lightgbm as lgb
import pandas as pd
import numpy as np
import tuning_shot as tst
import sklearn
import joblib
import test_analysis_shot as tas
from sklearn import metrics
from CORAL import CORAL
from JDA import JDA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def batch_cov(Xs, Xt):
    coral = CORAL()
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    Xs_new = np.empty([0, Xs.shape[1]])
    for i in range(int(np.round(ns / nt))):
        print('cov:{}'.format(i))
        if nt * (i + 1) < ns:
            Xs_cov, A = coral.fit(Xs[nt * i:nt * (i + 1), :], Xt)
        else:
            Xs_cov, A = coral.fit(Xs[nt * i:, :], Xt)
        Xs_new = np.append(Xs_new, Xs_cov, axis=0)
    return Xs_new


def coral_label(Xsdf, Xtdf):
    coral = CORAL()
    disdata_s = Xsdf.loc[Xsdf['disrup_tag'] == 1]
    undisdata_s = Xsdf.loc[Xsdf['disrup_tag'] == 0]
    disdata_t = Xtdf.loc[Xtdf['disrup_tag'] == 1]
    undisdata_t = Xtdf.loc[Xtdf['disrup_tag'] == 0]
    undisdata_s_sample = undisdata_s.sample(frac=0.2, random_state=1)
    undisdata_t_sample = undisdata_t.sample(frac=0.5, random_state=1)
    X_dis_s = disdata_s.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    X_dis_t = disdata_t.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    X_undis_s = undisdata_s.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    X_undis_s_sample = undisdata_s_sample.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    X_undis_t_sample = undisdata_t_sample.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

    Xs_dis_new, A_dis = coral.fit(X_dis_s, X_dis_t)
    Xs_cov, A_undis = coral.fit(X_undis_s_sample, X_undis_t_sample)
    Xs_undis_new = np.real(np.dot(X_undis_s, A_undis))
    Xs_new = np.append(Xs_dis_new, Xs_undis_new, axis=0)
    return Xs_new


def jda_con(Xs, Ys, Xt, Yt):
    jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1, T=10)
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    Xs_new = np.empty([0, Xs.shape[1]])
    for i in range(int(np.round(ns / nt))):
        print('cov:{}'.format(i))
        if nt * (i + 1) < ns:
            Xs_cov, Xt_cov = jda.jda_fit(Xs[nt * i:nt * (i + 1), :], Ys[nt * i:nt * (i + 1)], Xt, Yt)
        else:
            Xs_cov, Xt_cov = jda.jda_fit(Xs[nt * i:, :], Ys[nt * i:], Xt, Yt)
        Xs_new = np.append(Xs_new, Xs_cov, axis=0)
    return Xs_new


def coral_inc(Xs, Ys, Xt, Yt, X_val, y_val):
    coral = CORAL()
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    for i in range(int(np.round(ns / nt))):
        print('cov:{}'.format(i))
        if nt * (i + 1) < ns:
            Xs_cov = coral.fit(Xs[nt * i:nt * (i + 1), :], Xt)
            Y_s = Ys[nt * i:nt * (i + 1)]
        else:
            Xs_cov = coral.fit(Xs[nt * i:, :], Xt)
            Y_s = Ys[nt * i:]
        Xs_new = np.append(Xs_cov, Xt, axis=0)
        ys_new = np.append(Y_s, Yt, axis=0)
        lgb_train = lgb.Dataset(Xs_new, ys_new)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': {'auc'}, 'learning_rate': 0.1,
                  'is_unbalance': True, 'min_data_in_leaf': 10,
                  'num_leaves': 10,
                  'max_depth': 3}

        evals_result = {}  # to record eval results for plotting

        print('Starting training...')
        # train
        if i == 0:
            a = None
        else:
            a = gbm
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=300,
                        valid_sets={lgb_train, lgb_eval},
                        evals_result=evals_result,
                        verbose_eval=100,
                        init_model=a,
                        keep_training_booster=True,
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


if __name__ == '__main__':
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    X_t = df_H_beta_mix.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_train = df_train['disrup_tag'].values
    y_t = df_H_beta_mix['disrup_tag'].values
    # X_train = batch_cov(X_train, X_t)  # CORAL方法
    X_train = coral_label(df_train, df_H_beta_mix)  # dropout,破裂非破裂分开变换
    # X_train = jda_con(X_train, y_train, X_t, y_t)  # JDA方法
    Xs_new = np.append(X_train, X_t, axis=0)
    ys_new = np.append(y_train, y_t, axis=0)

    y_val = df_val['disrup_tag'].values
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

    # coral_inc(X_train, y_train, X_t, y_t, X_val, y_val)

    lgb_train = lgb.Dataset(Xs_new, ys_new)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': {'auc'}, 'learning_rate': 0.1,
              'is_unbalance': True, 'min_data_in_leaf': 10,
              'num_leaves': 10,
             }

    params_test1 = {'num_leaves': range(50, 100, 5)}

    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=300,
    #                 valid_sets={lgb_train, lgb_eval},
    #                 evals_result=evals_result,
    #                 verbose_eval=100,
    #                 early_stopping_rounds=30)
    gbm = lgb.LGBMClassifier(objective='binary',
                             is_unbalance=True,
                             metric='auc',
                             num_leaves=40,
                             learning_rate=0.1,
                             n_estimators=200,
                             )
    gsearch1 = GridSearchCV(estimator=gbm, param_grid=params_test1, scoring='roc_auc', cv=5,
                            verbose=1)
    gsearch1.fit(Xs_new, ys_new)
    gsearch1.best_params_, gsearch1.best_score_
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
