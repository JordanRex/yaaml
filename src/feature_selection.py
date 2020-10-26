# FEATURE SELECTION
#- near zero variance columns are removed (threshold=0.1)

from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier


class feat_selection():

    def __init__(self, train, valid, y_train, t=0.2):
        X, Y = self.variance_threshold_selector(train, valid, threshold=t)
        X, Y = self.rfecv(train=X, valid=Y, y_train=y_train)
        return X, Y

    # removing near zero variance columns
    def variance_threshold_selector(self, train, valid, threshold):
        print('Feature selection ...\n')
        print('input data shape is: ', train.shape, '\n')
        selector = VarianceThreshold(threshold)
        selector.fit(train)
        X = train[train.columns[selector.get_support(indices=True)]]
        Y = valid[valid.columns[selector.get_support(indices=True)]]
        print('output data shape is: ', X.shape, '\n')
        self.var_threshold=threshold
        return X, Y

    # using RFECV
    def rfecv(self, train, valid, y_train):
        # Create the RFE object and compute a cross-validated score.
        model = RandomForestClassifier(max_depth=7, max_features=0.25, n_estimators=100, n_jobs=-1)
        rfecv = RFECV(estimator=model, step=1, scoring='roc_auc', verbose=True)
        rfecv.fit(train, y_train)
        print("Optimal number of features : %d" % rfecv.n_features_, '\n')

        # Plot number of features VS. cross-validation scores
#         plt.figure()
#         plt.xlabel("Number of features selected")
#         plt.ylabel("Cross validation score (roc-auc)")
#         plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#         plt.show()

        features = [f for f, s in zip(train.columns, rfecv.support_) if s]
        train = train[features]
        valid = valid[features]
        self.final_features = features
        print('\n')
        return train, valid
