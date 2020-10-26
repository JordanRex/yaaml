#### NLP MODULE ####

"""
Description - Creates NLP features from the text columns in data (if any) and adds them to the main model (not automated, user discretion). Makes a large number of word features (count, term frequency and hashed), performs PCA on them to reduce the dimension, and creates a separate model to validate the predictive power in the text features. If results are better than the baseline, an ensemble model is automatically created. the user is advised to experiment by adding the features to the main model and compare with the ensemble results. if the results are around the ballpark of the ensemble results, it is highly recommended to use them as features instead of as a separate model in an ensemble
"""

## NLP Class

class nlp_feats():

    def __init__(self, train, test, response, text_cols):
        x, X, y, Y = self.prepare(train, test, response, text_cols)
        x, X = self.make_feats(x, X)
        clf = self.model(x, X, y, Y)
        pred = clf.predict_proba(X)[:,1]

    def prepare(self, train, test, response, text_cols):
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
