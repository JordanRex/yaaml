# HELPER FUNCTIONS CLASS
from collections import Counter


class helper_funcs:

    def __init__(self):
        """ helper functions used across the pipeline """

    # find and append multiple dataframes of the type specified in string
    def append_datasets(self, cols_to_remove, string=['train', 'valid']):
        # pass either train or valid as str argument
        temp_files = [name for name in os.listdir('../input/') if name.startswith(string)]
        temp_dict = {}
        for i in temp_files:
            df_name = re.sub(string=i, pattern='.csv', repl='')
            temp_dict[df_name] = pd.read_csv(str('../input/' + str(i)), na_values=['No Data', ' ', 'UNKNOWN'])
            temp_dict[df_name].columns = map(str.lower, temp_dict[df_name].columns)
            temp_dict[df_name].drop(cols_to_remove, axis=1, inplace=True)
            chars_to_remove = [' ', '.', '(', ')', '__', '-']
            for j in chars_to_remove:
                temp_dict[df_name].columns = temp_dict[df_name].columns.str.strip().str.lower().str.replace(j, '_')
        temp_list = [v for k, v in temp_dict.items()]
        if len(temp_list) > 1:
            temp = pd.concat(temp_list, axis=0, sort=True, ignore_index=True)
        else:
            temp = temp_list[0]
        return temp

    # datetime feature engineering
    def datetime_feats(self, train, valid):
        cols = [s for s in train.columns.values if 'date' in s]
        print('datetime feature engineering is happening ...', '\n')
        
        # nested function to derive the various datetime features for a given date column
        def dt_feats(df, col):
            df[col] = pd.to_datetime(df[i])
            #df[str(col+'_'+'day')] = df[col].dt.day
            df[str(col+'_'+'day_name')] = df[col].dt.day_name
            #df[str(col+'_'+'dayofweek')] = df[col].dt.dayofweek
            df[str(col+'_'+'dayofyear')] = df[col].dt.dayofyear
            #df[str(col+'_'+'days_in_month')] = df[col].dt.days_in_month
            #df[str(col+'_'+'month')] = df[col].dt.month
            df[str(col+'_'+'month_name')] = df[col].dt.month_name
            df[str(col+'_'+'quarter')] = df[col].dt.quarter
            df[str(col+'_'+'week')] = df[col].dt.week
            #df[str(col+'_'+'weekday')] = df[col].dt.weekday
            df[str(col+'_'+'year')] = df[col].dt.year
            #df[col] = df[col].dt.date
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
        global iter
        iter = 0
        for i in tqdm(cat_columns):
            grouped_df = pd.DataFrame(train_df.groupby([i])[response].agg(['mean', 'std'])).reset_index()
            grouped_df.rename(columns={'mean': str('mean_' + cat_columns[iter]),
                                       'std': str('std_' + cat_columns[iter])}, inplace=True)
            train_df = pd.merge(train_df, grouped_df, how='left')
            valid_df = pd.merge(valid_df, grouped_df, how='left')
            iter += 1
        return train_df, valid_df
    
    # for reading files
    def csv_read(self, file_path, cols_to_keep=None, dtype=None):
        self.cols_to_keep = cols_to_keep
        if dtype is None:
          x=pd.read_csv(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], encoding='latin-1', low_memory=False)
        else:
          x=pd.read_csv(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], encoding='latin-1', low_memory=False, dtype=dtype)
        chars_to_remove = [' ', '.', '(', ')', '__', '-', '/', '\'']
        for i in chars_to_remove:
            x.columns = x.columns.str.strip().str.lower().str.replace(i, '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        x.drop_duplicates(inplace=True)
        print(x.shape)
        return x
    
    def txt_read(self, file_path, cols_to_keep=None, sep='|', skiprows=1, dtype=None):
        # currently only supports salary files with the default values (need to implement dynamic programming for any generic txt)
        self.cols_to_keep = cols_to_keep
        if dtype is None:
          x=pd.read_table(file_path, sep=sep, skiprows=skiprows, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'])
        else:
          x=pd.read_table(file_path, sep=sep, skiprows=skiprows, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], dtype=dtype)
        chars_to_remove = [' ', '.', '(', ')', '__', '-', '/', '\'']
        for i in chars_to_remove:
            x.columns = x.columns.str.strip().str.lower().str.replace(i, '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        x.drop_duplicates(inplace=True)
        print(x.shape)
        return x

    def xlsx_read(self, file_path, cols_to_keep=None, sheet_name=0, dtype=None):
        self.cols_to_keep = cols_to_keep
        if dtype is None:
          x=pd.read_excel(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], sheet_name=sheet_name)
        else:
          x=pd.read_excel(file_path, na_values=['No Data', ' ', 'UNKNOWN', '', 'Not Rated', 'Not Applicable'], sheet_name=sheet_name, dtype=dtype)
        chars_to_remove = [' ', '.', '(', ')', '__', '-', '/', '\'']
        for i in chars_to_remove:
            x.columns = x.columns.str.strip().str.lower().str.replace(i, '_')
        if cols_to_keep is not None: x = x[cols_to_keep]
        x.drop_duplicates(inplace=True)
        print(x.shape)
        return x
    
    def process_columns(self, df, cols=None):
        if cols is None:
            df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('\s+|\s', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('[^\w+\s+]', '_', regex=True) if (x.dtype == 'object') else x)
            df = df.apply(lambda x: x.str.replace('\_+', '_', regex=True) if (x.dtype == 'object') else x)
        else:
            df = df.apply(lambda x: x.str.lower() if x.name in cols else x)
            df = df.apply(lambda x: x.str.strip() if x.name in cols else x)
            df = df.apply(lambda x: x.str.replace('\s+|\s', '_', regex=True) if x.name in cols else x)
            df = df.apply(lambda x: x.str.replace('[^\w+\s+]', '_', regex=True) if x.name in cols else x)
            df = df.apply(lambda x: x.str.replace('\_+', '_', regex=True) if x.name in cols else x)
        return df
  
    def nlp_process_columns(self, df, nlp_cols):
        df = df.apply(lambda x: x.str.replace('_', ' ') if x.name in nlp_cols else x)
        df = df.apply(lambda x: x.str.replace('\s+', ' ', regex=True) if x.name in nlp_cols else x)
        df = df.apply(lambda x: x.str.replace('crft', 'craft') if x.name in nlp_cols else x)
        return df