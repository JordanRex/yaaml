class yaamlFeatureEngineering:
    def __init__(self, params):
        self.params = params
        self.feat_sel = None
        self.encoders = None
        self.correlation_dropped = None
        self.to_drop = None

    def fit(self, train):
        # ENCODING
        categ_columns = train.select_dtypes(include=['object']).columns.values
        num_cols = list(set(train.columns)-set(categ_columns))
        train_cat = train[categ_columns]
        train_num = train[num_cols]

        # initialize and fit the encoders
        self.encoders = categ_encoders.encoding(train_cat, train_num, which=self.params['encoder'])

        # CORRELATION ANALYSIS
        # Create correlation matrix
        corr_matrix = train.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.75
        self.to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

        # FEATURE SELECTION
        if self.params['feat_selection'] == 'true':
            # initialize and fit the feature selector
            self.feat_sel = feat_selection.rfecv(train=train)

        return self

    def transform(self, data):
        # Apply the encoding
        categ_columns = data.select_dtypes(include=['object']).columns.values
        num_cols = list(set(data.columns)-set(categ_columns))
        data_cat = data[categ_columns]
        data_num = data[num_cols]

        data_cat, data_num = self.encoders.transform(data_cat, data_num)
        data = pd.concat([data_cat.reset_index(drop=True), data_num], axis=1)

        # Apply the correlation analysis
        data.drop(self.to_drop, axis=1, inplace=True)

        # Apply the feature selection
        if self.params['feat_selection'] == 'true':
            data = self.feat_sel.transform(data)

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
