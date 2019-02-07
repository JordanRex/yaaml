## for outlier identification and retraining for multiclass classification

# outlier class

from pandas_ml import ConfusionMatrix


class outlier():
    
    def __init__(self, train, valid, ytrain, yvalid):
        self.train = train
        self.ytrain = ytrain
        self.valid = valid
        self.yvalid = yvalid
        
        self.main()
        
    def main(self):
        mod = ExtraTreesClassifier(n_estimators=10, max_depth=10, n_jobs=-1, max_features=15,
                           class_weight={0:2, 1:2, 2:1, 3:1, 4:2, 5:2}, random_state=1)
        mod.fit(self.train, self.ytrain)
        print('validation score: ', mod.score(self.valid, self.yvalid))
        print('insample score: ', mod.score(self.train, self.ytrain))
        
        ## to get class vs class metrics
        train_pred = mod.predict(self.train)
        valid_pred = mod.predict(self.valid)
        train_preddf = pd.DataFrame({'ytrain': self.ytrain.ravel(), 'pred': train_pred.ravel()})
        train_preddf['outlier_flag1'] = np.where(abs(train_preddf['ytrain']-train_preddf['pred'])>=2.0, 1, 0)
        train_preddf['outlier_flag2'] = np.where(abs(train_preddf['ytrain']-train_preddf['pred'])>2.0, 1, 0)
        
        # fetch the indices of the strict and relaxed outliers
        indices_strict = train_preddf.loc[:, 'outlier_flag1']==0
        indices_conservative = train_preddf.loc[:, 'outlier_flag2']==0
        ixc = indices_conservative[indices_conservative].index
        ixs = indices_strict[indices_strict].index
        self.train_new = self.train.iloc[ixs,:]
        self.ytrain_new = self.ytrain[ixs]

        print('\n The train, train_new, ytrain, ytrain_new shapes are below:')
        print('', self.train.shape, '\n', self.train_new.shape, '\n', self.ytrain.shape, '\n', self.ytrain_new.shape, '\n')
        
        # train a new model on the outlier removed training frame
        mod_new = ExtraTreesClassifier(n_estimators=10, max_depth=10, n_jobs=-1, max_features=15,
                           class_weight={0:2, 1:2, 2:1, 3:1, 4:2, 5:2}, random_state=1)
        mod_new.fit(self.train_new, self.ytrain_new)
        print('new validation score :', mod_new.score(self.valid, self.yvalid))
        print('new insample score for train_new: ', mod_new.score(self.train_new, self.ytrain_new))
        print('new insample score for train_old: ', mod_new.score(train, ytrain))
        # new predictions
        valid_new_pred = mod_new.predict(self.valid)
        
        self.confmx_valid_old = ConfusionMatrix(y_true=self.yvalid, y_pred=valid_pred)
        self.confmx_valid_new = ConfusionMatrix(y_true=self.yvalid, y_pred=valid_new_pred)
        return None
    
outlier_mod = outlier(train, valid, ytrain, yvalid)