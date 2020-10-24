# HELPER FUNCTIONS CLASS
from collections import Counter


class helper_functions_class():

    def __init__(self):
        """ helper functions used across the pipeline """

    # datetime feature engineering
    def datetime_feats(self, train, valid):
        cols = [s for s in train.columns.values if 'date' in s]
        print('datetime feature engineering is happening ...', '\n')
        
        # nested function to derive the various datetime features for a given date column
        def dt_feats(df, col):
            df[col] = pd.to_datetime(df[i])
            df[str(col+'_'+'dayofyear')] = df[col].dt.dayofyear
            df[str(col+'_'+'week')] = df[col].dt.week
            df[str(col+'_'+'month')] = df[col].dt.month
            df[str(col+'_'+'quarter')] = df[col].dt.quarter
            df[str(col+'_'+'year')] = df[col].dt.year
            df = df.drop([col], axis=1)
            return df
        
        # loop function over all raw date columns
        for i in cols:
            train = dt_feats(train, i)
            valid = dt_feats(valid, i)
        return train, valid

    # function to get frequency count of elements in a vector/list
    def freq_count(self, input_vector):
        return Counter(input_vector)

    # function to make deviation encoding features
    def categ_feat_eng(self, train_df, valid_df, cat_columns, response):
        print('categorical feature engineering is happening ...', '\n')
        iter=0
        for i in tqdm(cat_columns):
            grouped_df = pd.DataFrame(train_df.groupby([i])[response].agg(['mean', 'std'])).reset_index()
            grouped_df.rename(columns={'mean': str('mean_' + cat_columns[iter]),
                                       'std': str('std_' + cat_columns[iter])}, inplace=True)
            train_df = pd.merge(train_df, grouped_df, how='left')
            valid_df = pd.merge(valid_df, grouped_df, how='left')
            iter+=1
        return train_df, valid_df
    
    def process_columns(self, df, cols=None):
        if cols is None:
            df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace(r'\s+|\s', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace(r'[^\w+\s+]', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace(r'\_+', '_', regex=True) if (x.dtype == 'object') else x)
        else:
            df[cols] = df[cols].apply(lambda x: x.str.lower())
            df[cols] = df[cols].apply(lambda x: x.str.strip())
            df[cols] = df[cols].apply(lambda x: x.str.replace(r'\s+|\s', '_', regex=True))
            df[cols] = df[cols].apply(lambda x: x.str.replace(r'[^\w\s]+', '_', regex=True))
            df[cols] = df[cols].apply(lambda x: x.str.replace(r'\_+', '_', regex=True))
        return df
  
    def nlp_process_columns(self, df, nlp_cols):
        df[nlp_cols] = df[nlp_cols].apply(lambda x: x.str.replace('_', ' '))
        df[nlp_cols] = df[nlp_cols].apply(lambda x: x.str.replace(r'\s+', ' ', regex=True))
        return df
    
    def retrieve_name(self, var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
            
    def getduplicates(self, df, idcol):
        return pd.concat(g for _, g in df.groupby(idcol) if len(g) > 1)

    def group_and_get_missingcount(self, df, grp_cols, missingcount_col):
        return df.groupby(grp_cols)[missingcount_col].apply(lambda x: x.isna().sum()/len(x)*100)
