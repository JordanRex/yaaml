# yaaml __init__.py

# importing packages
import os as os
# the base packages
import collections  # for the Counter function
import csv  # for reading/writing csv files
import pandas as pd, numpy as np, time as time, gc as gc, bisect as bisect, re as re
import datetime as dt

# Evaluation of the model
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, confusion_matrix, f1_score
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics, preprocessing
from sklearn.base import TransformerMixin
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# hyperopt modules
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from hyperopt import hp, tpe, STATUS_OK, fmin, Trials, space_eval
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

# modelling/clustering algorithms
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.covariance import EllipticEnvelope
# from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM

# main modules from root same directory
import helper_funcs as helpers
import miss_imputation as missimp
import encoding as encoders
import feature_engineering as feateng
import feature_selection as featsel
from sampling import sampler

# call the main script
import main

__version__ = '0.0.5'
__author__ = 'varunrajan'
__name__ = 'yaaml'
__org__ = '...'
