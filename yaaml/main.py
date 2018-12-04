"""####################################################################################################
Author - Varun Rajan
Package - yaaml 0.0.1
Pkg Description - Personal ML studio based on h2o automl and wrapped sklearn algos
Script - main.py
Script Description - The main script that is called
####################################################################################################"""


##################
# INITIALIZATION #
##################

## clear the workspace
%reset -f

## importing packages
import h2o
from h2o.automl import H2OAutoML

# define the global variables to be used later
MAX_EVALS = 10 # number of iterations/parameter sets created towards tuning
N_FOLDS = 5 # number of cv folds
randomseed = 1 # the value for the random state used at various points in the pipeline


##################################################################################################################################
## h2o AUTO_ML grid search framework
##################################################################################################################################
class h2o_automl:
    
    def __init__(self):
        """ main pipeline """
        return None
        
    def automl(self, os, us, bc, iter, response='response'):
        # initializing the h2o cluster
        h2o.init()
        # Import a sample binary outcome train/test set into H2O
        h2o_train = h2o.import_file(str(str(iter) + '_t_h2o.csv'), header=1)
        h2o_valid = h2o.import_file(str(str(iter) + '_v_h2o.csv'), header=1)
        # Identify the response and set of predictors
        x = list(h2o_train.columns)  #if x is defined as all columns except the response, then x is not required
        x.remove(response)
        # For binary classification, response should be a factor
        h2o_train[response] = h2o_train[response].asfactor()
        h2o_valid[response] = h2o_valid[response].asfactor()

        randomseed = 1
        aml = H2OAutoML(max_runtime_secs = 300, stopping_metric='mean_per_class_error', sort_metric='mean_per_class_error',
                        class_sampling_factors=[os, us], balance_classes = bc)
        aml.train(y = 'response', training_frame = h2o_train)
        
        # Print Leaderboard (ranked by xval metrics)
        print(aml.leaderboard)
        # Evaluate performance on a test set
        perf = aml.leader.model_performance(h2o_valid)
        print('The validation performance (auc) is ', perf.auc())
        return aml, h2o_valid
        
    def get_score(self, aml, h2o_valid, y_valid, threshold = 0.1):
        pred2 = aml.predict(h2o_valid)[:,2]
        pred = pred2.as_data_frame().as_matrix()
        predict = np.where(pred > threshold, 1, 0)
        y_test=y_valid

        recall_score = sklearn.metrics.recall_score(y_pred=predict, y_true=y_test)
        precision_score = sklearn.metrics.precision_score(y_pred=predict, y_true=y_test)
        f1_score = sklearn.metrics.f1_score(y_pred=predict, y_true=y_test)
        auc_score = roc_auc_score(y_test, pred)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_pred=predict, y_true=y_test).ravel()
        print(sklearn.metrics.confusion_matrix(y_pred=predict, y_true=y_test), '\n')
        print('recall score is: ', recall_score)
        print('precision score is: ', precision_score)
        print('f1_score is: ', f1_score)
        print('accuracy score: ', sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predict))
        print('The final AUC after taking the best params and num_rounds when it stopped is {:.4f}.'.format(auc_score), '\n')
        
        return pred2, predict, auc_score
    
##################################################################################################################################
## MOTHER OF ALL PIPELINES - hyperopt based ##
##################################################################################################################################
class main:
    
    def __init__(self):
        """ random """
        return None    
    
    def prepare(self, cols_to_remove, response='label'):
        """ checks first if backup pickles exist in folder already. 
        if so skips the major computation segment that is already back up """
        
        pickle_files = [x for x in os.listdir() if re.search(pattern='.pickle', string=x)]
        trials = {'backup_1': None}
        
        if len(pickle_files) > 0 :
            # opening the first pickle only for now (will later add a loop for all the pickle files)
            
            print('backups available...hence using them stead of reinventing the wheel \n')
            f = open(pickle_files[0], "rb")
            train = pickle.load(f)
            valid = pickle.load(f)
            y_train = pickle.load(f)
            y_valid = pickle.load(f)
            backup_md = pickle.load(f)
            f.close()
            print(backup_md)
            
            trials['backup_1'] = main.backup_optimize(train=train, valid=valid, y_train=y_train, 
                                                      y_valid=y_valid, backup_md=backup_md)
        
        else :
            # read in the train and validation datasets
            # clean column names and remove unwanted columns
            # append the (multiple?) train datasets into a single one (simple appending for now)

            print('1. Appending the multiple train/valid datasets in the working directory \n')
            train = helper_funcs.append_datasets(string='TRAIN', cols_to_remove=cols_to_remove)
            valid = helper_funcs.append_datasets(string='VALID', cols_to_remove=cols_to_remove)
            main.removed_cols = cols_to_remove ## attribute

            # creating the datetime features from date columns (works only for cols with date in header, modify for other cases)
            print('2. Datetime features are being created for the columns (which have "date" in their column name) \n')
            train, valid = helper_funcs.datetime_feats(train, valid)

            # missing value threshold control (for both rows and columns)
            mt = 0.5
            print(train.shape, '\n')
            train.dropna(thresh=mt*(train.shape[0]), axis=1, inplace = True)
            train.dropna(thresh=mt*(train.shape[1]), axis=0, inplace = True)
            print(train.shape, '\n')
            valid = valid[train.columns]
            valid.dropna(thresh=mt*(valid.shape[0]), axis=1, inplace = True)
            train = train[valid.columns]
            main.missing_threshold = mt ## attribute

            # reset the index since inplace operations happened earlier
            train.index = pd.RangeIndex(len(train.index))
            valid.index = pd.RangeIndex(len(valid.index))
            # save the global ids for mapping later (forward looking)
            valid_ids = valid[['original_id', response]]
            main.validation_labels = valid_ids ## attribute
            valid_ids.to_csv('test_dfs.csv', index=False)
            valid.drop('original_id', axis=1, inplace=True)
            train.drop('original_id', axis=1, inplace=True)
            X_train = pd.DataFrame(train)
            X_valid = pd.DataFrame(valid)
            # the class balance in the training dataset for the response
            print(helper_funcs.freq_count(X_train[response]), '\n')
            # creating the response vector
            y_train = X_train[response].values
            y_valid = X_valid[response].values

            # categorical columns (names, indices and dtypes)
            x = list(X_train.dtypes)
            x_1 = [1 if x == 'O' else 0 for x in x]
            categorical_idx = [i for i, x in enumerate(x_1) if x == 1]
            # Get feature names and their values for categorical data (needed for LIME)
            cat_columns = X_train.select_dtypes(include=['object']).columns.values
            X_train, X_valid = helper_funcs.categ_feat_eng(X_train, X_valid, cat_columns)

            # drop the response
            X_train = X_train.drop([response], axis = 1)
            X_valid = X_valid.drop([response], axis = 1)

            # call the main optimize function that does the whole tuning (inside the nested score function)
            trials = main.optimize(train=X_train, valid=X_valid, y_train=y_train, y_valid=y_valid)
        return trials
    
    # function to be minimized and sent to the optimize function of hyperopt
    def score(params):
        start_time = time.time()
        
        global ITERATION
        ITERATION += 1
        
        train=main.train
        valid=main.valid
        y_train=main.y_train
        y_valid=main.y_valid
        
        print('\n', params, '\n')
        #######################################################################################################
        ## ENCODING ##
        #######################################################################################################
        cat_columns = train.select_dtypes(include=['object']).columns.values
        train_cat = train[cat_columns]
        num_cols = list(set(train.columns)-set(train_cat.columns))
        train_num = train[num_cols]
        valid_cat = valid[cat_columns]
        valid_num = valid[num_cols]
        
        train_cat, valid_cat, categorical_names = categ_encoders.encoding(train_cat, valid_cat, y_train, y_valid,
                                                                            which=params['encoder'])
        train = pd.concat([train_cat.reset_index(drop=True), train_num], axis=1)
        valid = pd.concat([valid_cat.reset_index(drop=True), valid_num], axis=1)
        print('encoding completed ...', '\n')
        main.categorical_dict = categorical_names ## attribute
        #######################################################################################################

        
        #######################################################################################################
        ## CORRELATION ANALYSIS ##
        #######################################################################################################
        # remove highly correlated features to reduce further computation time
        print('correlation analysis is happening ...', '\n')
        # Create correlation matrix
        corr_matrix = train.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.75
        to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]        
        # Drop features
        #print(to_drop, '\n')
        train.drop(to_drop, axis=1, inplace=True)
        valid.drop(to_drop, axis=1, inplace=True)
        print('correlation analysis completed ...', '\n')
        main.cor_dropped_vars = to_drop ## attribute
        #######################################################################################################
    
        
        #######################################################################################################
        ## MISSING VALUE IMPUTATION ##
        #######################################################################################################
        # store all feature names
        feat_names = train.columns.values
        feat_names2 = valid.columns.values
        
        if params['miss_treatment'] == 'simple':
            miss_enc = DataFrameImputer()
            miss_enc.fit(X=train)
            train_new = miss_enc.transform(train)
            valid_new = miss_enc.transform(valid)
        elif params['miss_treatment'] in ['KNN', 'MICE']:
            train_new = DataFrameImputer.fancy_impute(train, which_method=params['miss_treatment'])
            valid_new = DataFrameImputer.fancy_impute(valid, which_method=params['miss_treatment'])

        # returning as pandas dataframes to retain feature names for LIME and feature importance plots
        train = pd.DataFrame(data=train_new, columns=feat_names)
        valid = pd.DataFrame(data=valid_new, columns=feat_names2)
        print('missing value treatment completed ...', '\n')
        #######################################################################################################
        

        #######################################################################################################
        ## STATUS REPORT ##
        #######################################################################################################
        print('STATUS REPORT \n')
        print(train.shape)
        print(valid.shape)
        print(y_train.shape)
        print(y_valid.shape)
        print(collections.Counter(y_train))
        print(collections.Counter(y_valid))
        #######################################################################################################
        
        #######################################################################################################
        ## FEATURE ENGINEERING ##
        #######################################################################################################
        """ the feature engineering module 
            - 1. PCA/ICA/TSVD/GRP/SRP
            - 2. KMEANS """
        
        ## 1.
        feat_eng_instance = feat_eng()
        feat_eng_instance.decomp_various(train, valid, n=int(params['decomp_feats']), which_method=params['scaler'])
        train, valid = feat_eng_instance.return_combined(train, valid)
        
        ## 2.
        train, valid = feat_eng.kmeans_feats(train_df=train, valid_df=valid, m=int(params['kmeans_n']))
        #######################################################################################################
        
        
        #######################################################################################################
        ## FEATURE SELECTION ##
        #######################################################################################################
        train, valid = feat_selection.variance_threshold_selector(train=train, valid=valid, threshold=0.1)
        
        if params['feat_selection'] == 'true':
            train, valid = feat_selection.rfecv(train=train, valid=valid, y_train=y_train)
        #######################################################################################################
        
        
        #######################################################################################################
        ## SAMPLING ##
        #######################################################################################################
        """ oversampling or undersampling or oversampling with undersampling """
        
        if params['sampler']['choice'] == 'yes':
            train, y_train = sampler(X_train=train, y_train=y_train, 
                                     which=params['sampler']['which_method'], 
                                     frac=params['sampler']['frac'])
        else :
            print('no sampling done in this pipeline', '\n')
        #######################################################################################################
        
        
        #######################################################################################################
        ## BACKUP ##
        #######################################################################################################
        backup = str(str(ITERATION) + str(dt.now().strftime('_%H_%M_%d_%m_%Y.pickle')))
        f = open(backup, "wb")
        pickle.dump(train, f)
        pickle.dump(valid, f)
        pickle.dump(y_train, f)
        pickle.dump(y_valid, f)
        
        backup_md = {'params': params, 'pickle_name': backup, 'randomseed': randomseed}
        pickle.dump(backup_md, f)
        
        f.close()
        #######################################################################################################
        
        
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
        
        aml, h2o_valid = h2o_automl.automl(os=h2o_os, bc=h2o_bc, us=h2o_us, iter=ITERATION)
        
        pred, predict, score = h2o_automl.get_score(aml=aml, h2o_valid=h2o_valid, y_valid=y_valid,
                                                    threshold=aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        
        setattr(main, str('aml_' + str(ITERATION)), aml)
        setattr(main, str('pred_' + str(ITERATION)), pred)
        setattr(main, str('predict_' + str(ITERATION)), predict)
        setattr(main, str('score_' + str(ITERATION)), score)
        setattr(main, str('threshold_' + str(ITERATION)), aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        setattr(main, str('h2o_valid_' + str(ITERATION)), h2o_valid)
        #######################################################################################################
        
        loss = 1 - score
        end_time = time.time()
        time_taken = timedelta(seconds = round(end_time - start_time))
        print("Execution took: %s secs (Wall clock time)" % time_taken)

        return {'loss': loss, 'status': STATUS_OK, 'params': params, 'auc': score, 'eval_time': time_taken}
    
    # function to do hyperparameter tuning with hyperopt (bayesian based method)
    def optimize(train, valid, y_train, y_valid):
                
        main.train = train
        main.valid = valid
        main.y_train = y_train
        main.y_valid= y_valid        
        
        # Keep track of evals
        global ITERATION
        ITERATION = 0
        global trials
        trials = Trials()
        
        # space to be traversed for the hyperopt function
        space = {
            'encoder': hp.choice('encoder', ['he', 'le']),
            'eval_time': time.time(),
            'miss_treatment': hp.choice('missing', ['simple']),
            'decomp_feats': hp.quniform('n', 2, 5, 1),
            'scaler': hp.choice('scaler', ['ss', 'mm']),
            'kmeans_n': hp.quniform('m', 2, 3, 1),
            'feat_selection': hp.choice('rfecv', ['false', 'false']),
            'sampler': hp.choice('sampler', [
                {
                    'choice': 'no',
                    'which_method': hp.choice('sampling', ['smote_enn']),
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
                    'balance_classes': hp.choice('bc', ['False', 'False'])
                }
            ])
        }
        
        best = fmin(main.score, space, algo=tpe.suggest, trials=trials, max_evals=MAX_EVALS,
                    rstate=np.random.RandomState(randomseed))
        best = trials.best_trial['result']['params']
        
        main.best = space_eval(space, trials.argmin)
        main.trials = trials
        
        return trials # results of all the iterations, the best params
    
    def backup_optimize(train, valid, y_train, y_valid, backup_md):
        main.train = train
        main.valid = valid
        main.y_train = y_train
        main.y_valid= y_valid
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
        
        aml, h2o_valid = h2o_automl.automl(os=h2o_os, bc=h2o_bc, us=h2o_us, iter=ITERATION)
        
        pred, predict, score = h2o_automl.get_score(aml=aml, h2o_valid=h2o_valid, y_valid=y_valid,
                                                    threshold=aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        
        setattr(main, str('aml_' + str(ITERATION)), aml)
        setattr(main, str('pred_' + str(ITERATION)), pred)
        setattr(main, str('predict_' + str(ITERATION)), predict)
        setattr(main, str('score_' + str(ITERATION)), score)
        setattr(main, str('threshold_' + str(ITERATION)), aml.leader.find_threshold_by_max_metric('min_per_class_accuracy'))
        setattr(main, str('h2o_valid_' + str(ITERATION)), h2o_valid)
        #######################################################################################################
        trials = score
        
        return trials
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############ FOR LATER ####################
# # create an instance of the main class and call it
# x = main()
# trials = x.prepare(cols_to_remove=['id'], response='response')
# # get scores from the best aml object for our validation set
# h2o_automl.get_score(aml=x.aml_1, h2o_valid=x.h2o_valid_1, y_valid=x.y_valid, threshold=0.5)
# # use below segment to read in any particular backup as needed (touchpoint is immediately prior to calling automl)
# import pickle
# f = open("backup_be_simple.pickle", "rb")
# train = pickle.load(f)
# valid = pickle.load(f)
# y_train = pickle.load(f)
# y_valid = pickle.load(f)
# f.close()
# h2o.cluster().shutdown()