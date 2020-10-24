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
import pandas as pd


class feature_engineering_class():

    def __init__(self, train, valid, scaler_method='ss', decomposition_methods=['PCA'], ):
        self.t = train
        self.v = valid
        self.scm = scaler_method
        self.dcm = decomposition_methods

    def scalers(self):
        """
        Standard scaler = 'ss'
        MinMax scaler = 'mm'
        """
        sc = StandardScaler() if self.scm == 'ss' else MinMaxScaler()
        sc.fit(self.t)
        return pd.DataFrame(sc.transform(self.t), columns=self.t.columns.values), pd.DataFrame(sc.transform(self.v), columns=self.v.columns.values)

    def decomp_various(self, n):
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
            t, v = self.scalers()

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

    def group_agg_feats(self, df, group_cols, agg_col, new_cols):
        df_grp = df.groupby(group_cols, group_keys=False).apply(lambda x: x.sample(frac=len(x) * .667)).reset_index()
        df_grp_new = df_grp.groupby(group_cols, as_index=False).agg({agg_col: ['sum', 'mean']}).my_flatten_cols()
        df_grp_new.columns = new_cols
        df = df.merge(df_grp_new, how='left')
        return df

    def group_cumsum_feats(self, df, group_cols, cumsum_cols, sort_col):
        df_temp = (df.
                   sort_values(sort_col).
                   groupby(group_cols, sort=False).
                   sum().
                   groupby(level=[0])[cumsum_cols].
                   cumsum().
                   reset_index().
                   my_flatten_cols())
        group_cols.extend(list(['cumsum_' + s for s in cumsum_cols]))
        df_temp.columns = group_cols
        df = df.merge(df_temp, how='left')
        return df

    def group_cummean_feats(self, df, group_cols, cummean_col, sort_col):
        for i in group_cols:
            df.sort_values([i, sort_col], ascending=[True, True], inplace=True)
            df.reset_index(drop=True, inplace=True)
            new_col = str('cummean_' + i + '_' + cummean_col)
            df[new_col] = df.groupby(i, sort=False)[cummean_col].expanding().mean().reset_index(drop=True)
        return df

    # below function to be tweaked. incorporate cummax/cummin version
    def group_topn_feats(self, df, group_cols, sort_col, subset_cols, new_cols, npwhere_col, npwhere_list, which, flag_col, n=1):
        # which = ['top', 'bottom']
        df[sort_col] = df[sort_col].astype(float)
        if which == 'top':
          # df['new'] = df.groupby('id').value.cummax()
            df_temp = df.groupby(group_cols)[sort_col].nlargest(n)
        elif which == 'bottom':
            df_temp = df.groupby(group_cols)[sort_col].nsmallest(n)
        df_temp = df[subset_cols].loc[df_temp.index.get_level_values(1)].reset_index(drop=True)
        df_temp.columns = new_cols
        df_temp[flag_col] = np.where(df_temp[npwhere_col].isin(npwhere_list), 1, 0)
        df_temp.drop(npwhere_col, inplace=True, axis=1)
        df = df.merge(df_temp, how='left')
        return df

    def date_converter(self, df):
        date_cols = [col for col in df.columns if 'date' in col]
        for i in date_cols:
            df[i] = df[i].astype(str)
            df[i] = df[i].str.replace(' +', ' ', regex=True)
            df[i] = df[i].str.split(' ').str[0].astype(str)
            df[i] = df[i].str.replace('[^\w\s]+', '_', regex=True)
            df[i + '_year'] = df[i].str.split('_').str[2].astype(str).apply(lambda s: ('20' + s if len(s) == 2 else s))
            df = df.loc[df[i + '_year'] != '0000']
            df[i + '_month'] = df[i].astype(str).str.split('_').str[0].astype(str).apply(lambda s: ('0' + s if len(s) == 1 else s))
            df[i + '_day'] = df[i].astype(str).str.split('_').str[1].astype(float)
            df[i + '_month'] = np.where(df[i + '_year'].astype(float).isin([99, 9999, 2099]), 12, df[i + '_month'])
            df[i + '_day'] = np.where(df[i + '_year'].astype(float).isin([99, 9999, 2099]), 31, df[i + '_day'])
            df[i + '_year'] = np.where(df[i + '_year'].astype(float).isin([99, 9999, 2099]), 2019, df[i+ '_year'])
            df[i + '_yearmonth'] = df[i + '_year'].map(str) + df[i+ '_month'].astype(str)
            df[i + '_year'] = df[i + '_year'].astype(float)
            df[i + '_month'] = df[i + '_month'].astype(float)
        return df

    def groupby_count(self, df, grp1_col, grp2_cols, count_col, level_two='no'):
        for i in grp2_cols:
            col_name = str('count_' + count_col)
            if level_two=='yes':
                col_name = 'level2_size'
            else:
                col_name = 'level1_size'
            grp_cols = [grp1_col, i]
            df2 = df.groupby(grp_cols)[count_col].nunique().to_frame(name=col_name).reset_index()
            df = df.merge(df2, how='left')
        return df

    def groupby_compare(self, df, grp_cols, grp_fixed_cols, transform_col, new_colname):
        for i in grp_cols:
            all_grp_cols = grp_fixed_cols.copy()
            if i is not None:
                all_grp_cols.extend([i])
            df1 = df.groupby(all_grp_cols)[transform_col].transform('size').sub(1)
            df2 = df.groupby(all_grp_cols)[transform_col].transform('sum').sub(df[transform_col])
            if i is not None:
                new_colname = str(new_colname + transform_col + '_atlevel_' + i)
            if i is None:
                new_colname = str(new_colname + transform_col + 'atlevel_', '...user_insert_here...')
            df[new_colname] = df2 / df1
            df[new_colname] = df[transform_col] / df[new_colname]
        return df

    def fillna_df(self, df, fill_cols, mode, grp_col=None, abs_value=None, ref_value_col=None, ref_value_col_frac=None):
        """ mode = ['simple_abs', 'simple_ref', 'adv_fill'] """
        for i in fill_cols:
            if mode == 'simple_abs':
                df[i] = df[i].astype(float)
                df[i].fillna(value=abs_value, inplace=True)
            elif mode == 'simple_ref':
                df[i].fillna(value=df[ref_value_col], inplace=True)
            elif mode == 'adv_fill':
                df[i] = df.groupby(grp_col)[i].transform(lambda x: x.ffill())
                df['flagna'] = df[i].isnull()
                df[i] = df.groupby(grp_col)[i].transform(lambda x: x.bfill())
                if ref_value_col is not None:
                    df[i] = np.where(df['flagna'] == 1, df[i] - ref_value_col_frac * df[i], df[i])
                if abs_value is not None:
                    df[i] = np.where(df['flagna'] == 1, df[i] - abs_value, df[i])
                df.drop('flagna', axis=1, inplace=True)
                df.reset_index(inplace=True, drop=True)
        return df

    def force_numeric(self, df, cols=None):
        if cols is None:
            allcols_exceptdates = df.select_dtypes(exclude=['datetime64']).columns.values
            df[allcols_exceptdates] = df[allcols_exceptdates].apply(pd.to_numeric, errors='coerce')
        else:
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        return df
