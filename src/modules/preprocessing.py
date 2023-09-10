class yaamlPreprocessing:
    def __init__(self, params, response_variable='response', idcol='id'):
        self.params = params
        self.response = response_variable
        self.idcol = idcol
        self.miss_enc = None
        self.le = None

    def fit(self, train):
        # creating the datetime features from date columns
        train = helper_funcs.datetime_feats(train)

        # missing value threshold control
        mt = 0.5
        train.dropna(thresh=mt*(train.shape[0]), axis=1, inplace=True)
        train.dropna(thresh=mt*(train.shape[1]), axis=0, inplace=True)

        # label encode if mode is classify
        ytrain = train[self.response].values
        self.le = LabelEncoder()
        self.le.fit(ytrain)

        # train the missing value imputer
        if self.params['miss_treatment'] == 'simple':
            self.miss_enc = DataFrameImputer()
            self.miss_enc.fit(X=train)

        return self

    def transform(self, data):
        # apply the datetime features creation
        data = helper_funcs.datetime_feats(data)

        # apply the missing value threshold control
        mt = 0.5
        data.dropna(thresh=mt*(data.shape[0]), axis=1, inplace=True)
        data.dropna(thresh=mt*(data.shape[1]), axis=0, inplace=True)

        # reset the index
        data.reset_index(inplace=True, drop=True)

        # drop the id column
        data.drop(self.idcol, axis=1, inplace=True)

        # apply the label encoding
        ydata = self.le.transform(data[self.response].values)

        # apply the missing value imputation
        if self.params['miss_treatment'] == 'simple':
            data_new = self.miss_enc.transform(data)
            data = pd.DataFrame(data=data_new, columns=data.columns.values)

        # drop the response
        data = data.drop([self.response], axis=1)

        return data, ydata

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
