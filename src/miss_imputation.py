## MISSING VALUE IMPUTATION CLASS ##

#- implements multiple methods to perform missing value treatment
#- simple methods (mean/median/mode) imputation for categorical and numerical features.
#    - slightly tweak for adding grouped level if needed
#    - currently built to perform imputation consistent to below process (after encoding)
#- fancy methods (NNM/KNN/MICE) imputation for numerical features only
#    - requires encoding to be done prior

from sklearn.impute import SimpleImputer


class DataFrameImputer():

    def __init__(self, train, test, method):
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_cols = list(df.select_dtypes(include=numerics).columns)

list(df.select_dtypes(['object']).columns)

    def simple(self, train, test):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

"""
need to insert pipeline and feature union here to orchestrate the whole thing
"""
