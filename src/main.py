"""####################################################################################################
Author - Varun Rajan
Package - yaaml 0.0.5
####################################################################################################"""

import os
import collections
import csv
import pandas as pd
import numpy as np
import time
import gc
import bisect
import re
import datetime

from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, confusion_matrix, f1_score
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics, preprocessing
from sklearn.base import TransformerMixin
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from hyperopt import hp, tpe, STATUS_OK, fmin, Trials, space_eval
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

from . import helper_funcs as helpers
from . import miss_imputation as missimp
from . import encoding as encoders
from . import feature_engineering as feateng
from . import feature_selection as featsel
from .sampling import sampler

import h2o
from h2o.automl import H2OAutoML

##################################################################################################################################
## MOTHER OF ALL PIPELINES - hyperopt based ##
##################################################################################################################################


class yaamlAutoML:
    def __init__(
        self,
        data,
        preprocessing_params,
        feature_engineering_params,
        modeling_params
    ):
        self.preprocessing = yaamlPreprocessing(preprocessing_params)
        self.feature_engineering = yaamlFeatureEngineering(
            feature_engineering_params)
        self.modeling = yaamlModeling(modeling_params)

    def fit(self, data):
        data = self.preprocessing.fit_transform(data)
        data = self.feature_engineering.fit_transform(data)
        self.modeling.fit(data)

    def predict(self, data):
        data = self.preprocessing.transform(data)
        data = self.feature_engineering.transform(data)
        return self.modeling.predict(data)

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        return self.modeling.evaluate(predictions, labels)


class yaamlAutoML():

    def __init__(self,
                 train, valid,
                 maxevals=10, nfolds=3, randomseed=123,
                 response_variable='response', idcol='id',
                 mode='classify',  # mode = ['classify', 'regress']
                 # user to supply -> encoding, missing value treatment, sampling
                 encoding='oe', misstreatment='simple',
                 cols_to_remove=None):
        """ initialize the main variables """
        # define the global variables to be used later
        self.MAX_EVALS = maxevals  # number of iterations/parameter sets created towards tuning
        self.N_FOLDS = nfolds  # number of cv folds
        # the value for the random state used at various points in the pipeline
        self.randomseed = randomseed

        self.train = train
        self.valid = valid
        self.idcol = idcol
        self.response = response_variable
        if cols_to_remove is not None:
            self.cols_to_remove = cols_to_remove

        # Start the H2O cluster (locally)
        h2o.init()

        #######################################################################################################
        # ENCODING
        #######################################################################################################
        xtrain_cat = xtrain[categ_columns]
        num_cols = list(set(xtrain.columns)-set(xtrain_cat.columns))
        xtrain_num = xtrain[num_cols]
        xvalid_cat = xvalid[categ_columns]
        xvalid_num = xvalid[num_cols]

        xtrain_cat, xvalid_cat, categorical_names = categ_encoders.encoding(train_cat, valid_cat, y_train, y_valid,
                                                                            which=params['encoder'])
        train = pd.concat(
            [train_cat.reset_index(drop=True), train_num], axis=1)
        valid = pd.concat(
            [valid_cat.reset_index(drop=True), valid_num], axis=1)
        print('encoding completed ...', '\n')
        main.categorical_dict = categorical_names
        #######################################################################################################
        #######################################################################################################
        # CORRELATION ANALYSIS
        #######################################################################################################
        # remove highly correlated features to reduce further computation time
        print('correlation analysis is happening ...', '\n')
        # Create correlation matrix
        corr_matrix = train.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.75
        to_drop = [column for column in upper.columns if any(
            upper[column] > 0.75)]
        # Drop features
        train.drop(to_drop, axis=1, inplace=True)
        valid.drop(to_drop, axis=1, inplace=True)
        print('correlation analysis completed ...', '\n')
        main.cor_dropped_vars = to_drop
        #######################################################################################################
        #######################################################################################################
        # MISSING VALUE IMPUTATION
        #######################################################################################################
        # store all feature names
        feat_names = train.columns.values
        feat_names2 = valid.columns.values

        if params['miss_treatment'] == 'simple':
            miss_enc = DataFrameImputer()
            miss_enc.fit(X=train)
            train_new = miss_enc.transform(train)
            valid_new = miss_enc.transform(valid)
        elif params['miss_treatment'] in ['KNN', 'IterativeImputer']:
            train_new = DataFrameImputer.fancy_impute(
                train, which_method=params['miss_treatment'])
            valid_new = DataFrameImputer.fancy_impute(
                valid, which_method=params['miss_treatment'])

        # returning as pandas dataframes to retain feature names for LIME and feature importance plots
        train = pd.DataFrame(data=train_new, columns=feat_names)
        valid = pd.DataFrame(data=valid_new, columns=feat_names2)
        print('missing value treatment completed ...', '\n')
        #######################################################################################################
        #######################################################################################################
        # STATUS REPORT
        #######################################################################################################
        print('STATUS REPORT \n')
        print(train.shape)
        print(valid.shape)
        print(y_train.shape)
        print(y_valid.shape)
        print(collections.Counter(y_train))
        print(collections.Counter(y_valid))
        print('STATUS REPORT END \n')
        #######################################################################################################
        #######################################################################################################
        # FEATURE ENGINEERING
        #######################################################################################################
        """ the feature engineering module
            - 1. PCA/ICA/TSVD/GRP/SRP
            - 2. KMEANS """

        feat_eng_instance = feat_eng()
        feat_eng_instance.decomp_various(train, valid, n=int(
            params['decomp_feats']), which_method=params['scaler'])
        train, valid = feat_eng_instance.return_combined(train, valid)

        train, valid = feat_eng.kmeans_feats(
            train_df=train, valid_df=valid, m=int(params['kmeans_n']))
        #######################################################################################################
        #######################################################################################################
        # FEATURE SELECTION
        #######################################################################################################
        train, valid = feat_selection.variance_threshold_selector(
            train=train, valid=valid, threshold=0.1)

        if params['feat_selection'] == 'true':
            train, valid = feat_selection.rfecv(
                train=train, valid=valid, y_train=y_train)
        #######################################################################################################
        #######################################################################################################
        # SAMPLING
        #######################################################################################################
        """ oversampling or undersampling or oversampling with undersampling """

        if params['sampler']['choice'] == 'yes':
            train, y_train = sampler(x_train=train, y_train=y_train,
                                     which=params['sampler']['which_method'],
                                     frac=params['sampler']['frac'])
        else:
            print('no sampling done in this pipeline', '\n')
        #######################################################################################################
        #######################################################################################################
        # BACKUP
        #######################################################################################################
        backup = str(str(ITERATION) +
                     str(dt.now().strftime('_%H_%M_%d_%m_%Y.pickle')))
        f = open(backup, "wb")
        pickle.dump(train, f)
        pickle.dump(valid, f)
        pickle.dump(y_train, f)
        pickle.dump(y_valid, f)

        backup_md = {'params': params, 'pickle_name': backup,
                     'randomseed': self.randomseed}
        pickle.dump(backup_md, f)

        f.close()
        #######################################################################################################
        #######################################################################################################
        # SAVE AS FLATFILES
        #######################################################################################################
        train['response'] = y_train
        valid['response'] = y_valid

        train.to_csv(str(str(ITERATION) + '_t_h2o.csv'), index=False)
        valid.to_csv(str(str(ITERATION) + '_v_h2o.csv'), index=False)
        #######################################################################################################
        #######################################################################################################
        # H2O AUTOML
        #######################################################################################################

        h2o_os = params['h2o_automl_params']['oversampling']
        h2o_us = params['h2o_automl_params']['undersampling']
        h2o_bc = params['h2o_automl_params']['balance_classes']

        aml, h2o_valid = Automl.automl(
            oversample=h2o_os, balanceclasses=h2o_bc, undersample=h2o_us, iter_value=ITERATION)

        pred, predict, score = Automl.get_score(aml=aml, h2o_valid=h2o_valid, y_valid=y_valid,
                                                threshold=aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))

        setattr(main, str('aml_' + str(ITERATION)), aml)
        setattr(main, str('pred_' + str(ITERATION)), pred)
        setattr(main, str('predict_' + str(ITERATION)), predict)
        setattr(main, str('score_' + str(ITERATION)), score)
        setattr(main, str('threshold_' + str(ITERATION)),
                aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        setattr(main, str('h2o_valid_' + str(ITERATION)), h2o_valid)
        #######################################################################################################

        loss = 1 - score
        end_time = time.time()
        time_taken = timedelta(seconds=round(end_time - start_time))
        print("Execution took: %s secs (Wall clock time)" % time_taken)

        return {'loss': loss, 'status': STATUS_OK, 'params': params, 'auc': score, 'eval_time': time_taken}

    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(self):
        # Keep track of evals
        global ITERATION
        ITERATION = 0
        global trials
        trials = Trials()

        # space to be traversed for the hyperopt function
        space = {
            'encoder': hp.choice('encoder', ['oe', 'he', 'ohe', 'be']),
            'eval_time': time.time(),
            'miss_treatment': hp.choice('missing', ['simple', 'KNN', 'IterativeImputer']),
            'decomp_feats': hp.quniform('n', 2, 5, 1),
            'scaler': hp.choice('scaler', ['ss', 'mm']),
            'kmeans_n': hp.quniform('m', 2, 3, 1),
            'feat_selection': hp.choice('rfecv', ['true', 'false']),
            'sampler': hp.choice('sampler', [
                {
                    'choice': 'yes',
                    'which_method': hp.choice('sampling', ['smote_enn', 'smote_tomek']),
                    'frac': hp.quniform('frac', 0.75, 1, 0.05)
                },
                {
                    'choice': 'no'
                }
            ]),
            'h2o_automl_params': hp.choice('sampling_params', [
                {
                    'undersampling': hp.uniform('us', 0.1, 1),
                    'oversampling': hp.uniform('os', 1, 5),
                    'balance_classes': hp.choice('bc', ['True', 'False'])
                }
            ])
        }

        best = fmin(main.score, space, algo=tpe.suggest, trials=trials, max_evals=self.MAX_EVALS,
                    rstate=np.random.RandomState(self.randomseed))
        best = trials.best_trial['result']['params']
        main.best = space_eval(space, trials.argmin)
        main.trials = trials
        return trials  # results of all the iterations, the best params

    def backup_optimize(train, valid, y_train, y_valid, backup_md):
        main.train = train
        main.valid = valid
        main.y_train = y_train
        main.y_valid = y_valid
        main.md = backup_md
        params = backup_md['params']

        # Keep track of evals
        global ITERATION
        ITERATION = backup_md['pickle_name'][0]

        #######################################################################################################
        ## SAVE AS FLATFILES ##
        #######################################################################################################
        train['response'] = y_train
        valid['response'] = y_valid

        train.to_csv(str(str(ITERATION) + '_t_h2o.csv'), index=False)
        valid.to_csv(str(str(ITERATION) + '_v_h2o.csv'), index=False)
        #######################################################################################################

        #######################################################################################################
        ## H2O AUTOML ##
        #######################################################################################################
        h2o_os = params['h2o_automl_params']['oversampling']
        h2o_us = params['h2o_automl_params']['undersampling']
        h2o_bc = params['h2o_automl_params']['balance_classes']

        aml, h2o_valid = Automl.automl(
            oversample=h2o_os, balanceclasses=h2o_bc, undersample=h2o_us, iter_value=ITERATION)

        pred, predict, score = Automl.get_score(aml=aml, h2o_valid=h2o_valid, y_valid=y_valid,
                                                threshold=aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))

        setattr(main, str('aml_' + str(ITERATION)), aml)
        setattr(main, str('pred_' + str(ITERATION)), pred)
        setattr(main, str('predict_' + str(ITERATION)), predict)
        setattr(main, str('score_' + str(ITERATION)), score)
        setattr(main, str('threshold_' + str(ITERATION)),
                aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        #######################################################################################################
        trials = score
        return trials


if __name__ == '__main__':
    yaamlClass()
