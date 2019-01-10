#### FEATURE ENGINEERING MODULE ####

#- Decomposition features
#    - PCA
#    - ICA
#    - TSVD
#    - GRP
#    - SRP
#    - ...
#- Clustering output feaatures
#    - KMeans
#    - ...
#- Deterministic features
#    - Binning
#    - ...
from sklearn.cluster import KMeans
import sklearn.decomposition as decomposition
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.random_projection as rp


class feat_eng(object):

    def __init__(self):
        """ this module contains several functions for creating new features. find below a brief description of each """

    def scalers(self, train, valid, which_method):
        if which_method == 'ss':
            sc = StandardScaler()
            sc.fit(train)
            train_new = pd.DataFrame(sc.transform(train), columns=train.columns.values)
            valid_new = pd.DataFrame(sc.transform(valid), columns=valid.columns.values)
            return train_new, valid_new # scale all variables to zero mean and unit variance, required for PCA and related
        if which_method == 'mm':
            mm = MinMaxScaler()
            mm.fit(train)
            train_new = pd.DataFrame(mm.transform(train), columns=train.columns.values)
            valid_new = pd.DataFrame(mm.transform(valid), columns=valid.columns.values)
            return train_new, valid_new # use this method to iterate

    def decomp_various(self, train, valid, n, which_method):
        global decomp_dfs
        decomp_dfs = {}
        decomp_methods = ['PCA', 'FastICA', 'TruncatedSVD', 'GaussianRandomProjection', 'SparseRandomProjection']

        for i in decomp_methods:
            if i == 'PCA':
                decomp_obj = getattr(decomposition, i)
                decomp_obj = decomp_obj(n_components=.75)
            elif i in ['FastICA', 'TruncatedSVD']:
                decomp_obj = getattr(decomposition, i)
                decomp_obj = decomp_obj(n_components=n)
            else:
                decomp_obj = getattr(rp, i)
                decomp_obj = decomp_obj(n_components=n, eps=0.3)

            # perform the multiple decomposition techniques
            train, valid = feat_eng.scalers(self, train, valid, which_method)

            decomp_obj.fit(train)
            decomp_train = pd.DataFrame(decomp_obj.transform(train))
            decomp_valid = pd.DataFrame(decomp_obj.transform(valid))
            cols = list(set(list(decomp_train)))
            cols = [str(i) + '_' + str(s) for s in cols]
            decomp_train.columns = cols
            decomp_valid.columns = cols

            decomp_dfs[i + '_train'] = decomp_train
            decomp_dfs[i + '_valid'] = decomp_valid

        feat_eng.df = decomp_dfs
        return None

    def return_combined(self, train, valid):
        #self.df

        for i in list(self.df.keys()):
            if bool(re.search('train', i)):
                train = pd.concat([train.reset_index(drop=True), self.df[i]], axis=1)
            else:
                valid = pd.concat([valid.reset_index(drop=True), self.df[i]], axis=1)
        return train, valid

    def kmeans_clusterer(train_df, valid_df, n):
        clusterer = KMeans(n, random_state=1, init='k-means++')
        # fit the clusterer
        clusterer.fit(train_df)
        train_clusters = clusterer.predict(train_df)
        valid_clusters = clusterer.predict(valid_df)
        return train_clusters, valid_clusters

    def kmeans_feats(train_df, valid_df, m=5):
        print('m is ', m, '\n')
        for i in range(2, m):
            t, v = feat_eng.kmeans_clusterer(train_df, valid_df, n=i)
            col_name = str('kmeans_' + str(i))
            t = pd.DataFrame({col_name: t})
            v = pd.DataFrame({col_name: v})
            train_df = pd.concat([train_df.reset_index(drop=True), t], axis=1)
            valid_df = pd.concat([valid_df.reset_index(drop=True), v], axis=1)
        return train_df, valid_df
