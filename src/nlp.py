#### NLP MODULE ####

"""
Description - Creates NLP features from the text columns in data (if any) and adds them to the main model (not automated, user discretion). Makes a large number of word features (count, term frequency and hashed), performs PCA on them to reduce the dimension, and creates a separate model to validate the predictive power in the text features. If results are better than the baseline, an ensemble model is automatically created. the user is advised to experiment by adding the features to the main model and compare with the ensemble results. if the results are around the ballpark of the ensemble results, it is highly recommended to use them as features instead of as a separate model in an ensemble
"""

## NLP Class

class nlp_feats():
    
    def __init__(self, response):
        return None
    
    def prepare(self, train_path = 'TRAIN.csv', test_path = 'VALID.csv'):
        train=pd.read_csv(train_path, na_values=['No Data', ' ', 'UNKNOWN'])
        test=pd.read_csv(test_path, na_values=['No Data', ' ', 'UNKNOWN'])
        text_cols = []
        train_text = train[text_cols]
        test_text = test[text_cols]
        train_text['string_all'] = train_text[text_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)
        test_text['string_all'] = test_text[text_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)
        train_text = train_text[['string_all']]
        test_text = test_text[['string_all']]
        y_train = train[[response]]
        y_test = test[[response]]
        return train_text, test_text, y_train, y_test
    
    def make_feats(self, train_text, test_text):
        # nlp feats

        ## HASHING ##
        # create the transform
        vectorizer = HashingVectorizer(n_features=500, ngram_range=(1,4), stop_words='english',
                                       strip_accents='unicode', analyzer='char', norm='l1')
        vectorizer.fit(train_text)
        train_new1 = pd.DataFrame(vectorizer.transform(train_text.string_all).todense())
        test_new1 = pd.DataFrame(vectorizer.transform(test_text.string_all).todense())

        ## COUNTVECTORIZER ##
        # create the transform
        vectorizer = CountVectorizer(strip_accents='unicode', ngram_range=(1,5), stop_words='english', max_features=100)
        # tokenize and build vocab
        vectorizer.fit(train_text.string_all)
        # summarize
        #print(vectorizer.vocabulary_)
        train_new2 = pd.DataFrame(vectorizer.transform(train_text.string_all).todense(), columns=vectorizer.get_feature_names())
        test_new2 = pd.DataFrame(vectorizer.transform(test_text.string_all).todense(), columns=vectorizer.get_feature_names())

        ## TFIDVECTORIZER ##
        # create the transform
        vectorizer = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,4), stop_words='english', max_features=100)
        # tokenize and build vocab
        vectorizer.fit(train_text.string_all)
        train_new3 = pd.DataFrame(vectorizer.transform(train_text.string_all).todense(), columns=vectorizer.get_feature_names())
        test_new3 = pd.DataFrame(vectorizer.transform(test_text.string_all).todense(), columns=vectorizer.get_feature_names())

        train_new = pd.concat([train_new1, train_new2, train_new3], ignore_index=True, axis=1)
        test_new = pd.concat([test_new1, test_new2, test_new3], ignore_index=True, axis=1)
        return train_new, test_new
    
    def model(self, train_new, test_new, y_train, y_valid, which='rf'):
        if which == 'rf':
            # rf classifier
            clf=RandomForestClassifier(max_depth=15, n_estimators=300, random_state=1)
            clf.fit(train_new, y_train)
        elif which == 'sgd':
            # sgd classifier
            clf=sklearn.linear_model.SGDClassifier(loss='log', penalty='l1', random_state=6)
            clf.fit(train_new, y_train)
        elif which == 'xgb':
            # xgb classifier
            clf=xgb.XGBClassifier(learning_rate=0.02, n_estimators=100, colsample_bytree=0.9, 
                              subsample=0.9, scale_pos_weight=1, max_depth=10)
            clf.fit(train_new, y_train)        
        return clf
    
    def main(self, w1):
        print('preparing ... \n')
        train_text, test_text, y_train, y_valid = self.prepare()
        print('making the features ... \n')
        train_new, test_new = self.make_feats(train_text, test_text)
        print('making the predictions ... \n')
        clf = self.model(train_new, test_new, y_train, y_valid)
        
        # get the h2o model predictions and make the simple weighted ensemble predictions
        h2o_model_predictions = w1.as_data_frame()

        return clf, test_new, y_valid, h2o_model_predictions
    
    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def best_thresh_score(self, yp, yt):
        rc = recall_score(y_pred=yp, y_true=yt)
        ac = accuracy_score(y_pred=yp, y_true=yt)
        rc_ac_flag = 0
        if (rc>0.6 and ac>0.6) : rc_ac_flag = 1
        score = (rc_ac_flag)*(0.6*rc + 0.4*ac)
        return score
    
    def opt_thresh(self, h2o_pred, nlp_pred, ens_pred):
        X = self.get_truncated_normal(mean=0.1, sd=0.2, low=0, upp=0.4)
        Y = list(X.rvs(1000))
        
        cols = ['thresh', 'recall', 'precision', 'f1', 'acc', 'score'] #score = (0.6*recall + 0.4*acc)
        thresh_grid_h2o, thresh_grid_nlp, thresh_grid_ens = [], [], []

        for i in Y:
            # for h2o
            h2o_predict=np.where(h2o_pred > i, 1, 0)
            thresh_grid_h2o.append([i, recall_score(y_pred=h2o_predict, y_true=y_valid),
                                precision_score(y_pred=h2o_predict, y_true=y_valid),
                                f1_score(y_pred=h2o_predict, y_true=y_valid),
                                accuracy_score(y_pred=h2o_predict, y_true=y_valid),
                               self.best_thresh_score(yp=h2o_predict, yt=y_valid)])
            
            # for nlp
            nlp_predict=np.where(nlp_pred > i, 1, 0)
            thresh_grid_nlp.append([i, recall_score(y_pred=nlp_predict, y_true=y_valid),
                                precision_score(y_pred=nlp_predict, y_true=y_valid),
                                f1_score(y_pred=nlp_predict, y_true=y_valid),
                                accuracy_score(y_pred=nlp_predict, y_true=y_valid),
                               self.best_thresh_score(yp=nlp_predict, yt=y_valid)])
            
            # for ensemble
            ens_predict=np.where(ens_pred > i, 1, 0)
            thresh_grid_ens.append([i, recall_score(y_pred=ens_predict, y_true=y_valid),
                                precision_score(y_pred=ens_predict, y_true=y_valid),
                                f1_score(y_pred=ens_predict, y_true=y_valid),
                                accuracy_score(y_pred=ens_predict, y_true=y_valid),
                               self.best_thresh_score(yp=ens_predict, yt=y_valid)])
            
        thresh_grid_h2o = pd.DataFrame(thresh_grid_h2o, columns=cols)
        thresh_grid_nlp = pd.DataFrame(thresh_grid_nlp, columns=cols)
        thresh_grid_ens = pd.DataFrame(thresh_grid_ens, columns=cols)
        
        thresh_grid_h2o.sort_values(by='score', ascending=False, inplace=True)
        thresh_grid_nlp.sort_values(by='score', ascending=False, inplace=True)
        thresh_grid_ens.sort_values(by='score', ascending=False, inplace=True)

        h2o_thresh = thresh_grid_h2o.reset_index(drop=True).iloc[0][0]
        nlp_thresh = thresh_grid_nlp.reset_index(drop=True).iloc[0][0]
        ens_thresh = thresh_grid_ens.reset_index(drop=True).iloc[0][0]
        
        return h2o_thresh, nlp_thresh, ens_thresh
        
    def predict(self, clf, test_new, y_valid, h2o_model_predictions, 
                h2o_thresh=0, nlp_thresh=0, ens_thresh=0, ens_weightage=0.9):
        h2o_pred = h2o_model_predictions.p1
        nlp_pred = clf.predict_proba(test_new)[:,1]
        ens_pred = (((1 - ens_weightage) * nlp_pred) + (ens_weightage * h2o_pred))
        
        if (h2o_thresh==0 and nlp_thresh==0 and ens_thresh==0):
            h2o_thresh, nlp_thresh, ens_thresh = self.opt_thresh(h2o_pred, nlp_pred, ens_pred)
        
        # h2o predictions summary
        print('h2o predictions summary \n')
        h2o_predict=np.where(h2o_pred > h2o_thresh, 1, 0)
        print('h2o model auc is: ', sklearn.metrics.roc_auc_score(y_score=h2o_pred, y_true=y_valid), '\n')
        print('h2o model recall is: ', sklearn.metrics.recall_score(y_pred=h2o_predict, y_true=y_valid), '\n')
        print('h2o model accuracy is: ', sklearn.metrics.accuracy_score(y_pred=h2o_predict, y_true=y_valid), '\n')
        print('h2o model precision is: ', sklearn.metrics.precision_score(y_pred=h2o_predict, y_true=y_valid), '\n')
        print('\n')
        
        # nlp predictions summary
        print('nlp predictions summary \n')
        nlp_predict=np.where(nlp_pred > nlp_thresh, 1, 0)
        print('nlp model auc is: ', sklearn.metrics.roc_auc_score(y_score=nlp_pred, y_true=y_valid), '\n')
        print('nlp model recall is: ', sklearn.metrics.recall_score(y_pred=nlp_predict, y_true=y_valid), '\n')
        print('nlp model accuracy is: ', sklearn.metrics.accuracy_score(y_pred=nlp_predict, y_true=y_valid), '\n')
        print('nlp model precision is: ', sklearn.metrics.precision_score(y_pred=nlp_predict, y_true=y_valid), '\n')
        print('\n')
        
        # ensemble predictions summary
        print('ensemble predictions summary \n')
        ens_predict=np.where(ens_pred > ens_thresh, 1, 0)
        print('ensemble model auc is: ', sklearn.metrics.roc_auc_score(y_score=ens_pred, y_true=y_valid), '\n')
        print('ensemble model recall is: ', sklearn.metrics.recall_score(y_pred=ens_predict, y_true=y_valid), '\n')
        print('ensemble model accuracy is: ', sklearn.metrics.accuracy_score(y_pred=ens_predict, y_true=y_valid), '\n')
        print('ensemble model precision is: ', sklearn.metrics.precision_score(y_pred=ens_predict, y_true=y_valid), '\n')
        
        return ens_pred
    
    
y=nlp_feats()

clf, test, yvalid, h2o_model_predictions = y.main(df)

ens_pred = y.predict(clf, test, yvalid, h2o_model_predictions, ens_weightage=0.95)