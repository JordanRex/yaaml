## MISSING VALUE IMPUTATION CLASS ##

#- implements multiple methods to perform missing value treatment
#- simple methods (mean/median/mode) imputation for categorical and numerical features.
#    - slightly tweak for adding grouped level if needed
#    - currently built to perform imputation consistent to below process (after encoding)
#- fancy methods (NNM/KNN/MICE) imputation for numerical features only
#    - requires encoding to be done prior

from sklearn.base import TransformerMixin
from fancyimpute import KNN, IterativeImputer, NuclearNormMinimization


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else
                               X[c].mean() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def num_missing(self):
        return sum(self.isnull())

    def imputer_method(self, column, method=['mean', 'median', 'most_frequent']):
        x = Imputer(missing_values = 'NaN', strategy=method, axis=0)
        return x.fit_transform(self[[column]]).ravel()

    def fancy_impute(self, X, Y=None, which_method='IterativeImputer'):
        """ currently supported algorithms are KNN, NNM and MICE from the fancyimpute package
        which_method = ['KNN', 'NNM', 'IterativeImputer']
        """
        print(which_method, ' based missing value imputation is happening ...', '\n')
        
        if which_method == 'NNM': X = NuclearNormMinimization().complete(X) # NNM method
        if which_method == 'KNN': X = KNN(k=5, verbose=False).complete(X) # KNN method
        
        if which_method == 'IterativeImputer':
            imputer = IterativeImputer()
            imputer.fit(X.values)
            X_new = pd.DataFrame(data=imputer.transform(X.values), columns=X.columns)
            Y_new = pd.DataFrame(data=imputer.transform(Y.values), columns=Y.columns)
        print('missing value imputation completed', '\n')
        return X_new, Y_new