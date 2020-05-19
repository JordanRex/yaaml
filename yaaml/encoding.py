# ENCODING CLASS

"""
using the category encoders package in python
refer url=https://contrib.scikit-learn.org/categorical-encoding/
"""

import category_encoders as ce


class categ_encoders():
    
    def  __init__(self):

        
    def encoding(train, valid, y_train, y_valid, which=['le', 'be', 'bne', 'ohe', 'he', 'oe']):
        if which == 'le':
            train, valid, categorical_names = categ_encoders.labelEncoder(train, valid)
        elif which in ['be', 'bne', 'ohe', 'he', 'oe']:
            train, valid, categorical_names = categ_encoders.ce_encodings(train, valid, y_train, y_valid, which)
        else :
            print('Not supported. Use one of [be, bne, he, oe, ohe]', '\n')
            exit()
        return train, valid, categorical_names

    def ce_encodings(train_df, valid_df, y_train, y_valid, encoding):
        print(str(encoding) + ' encoding is happening ...', '\n')
        if encoding=='bne':
            enc=ce.BaseNEncoder(base=3)
        elif encoding=='be':
            enc=ce.BinaryEncoder()
        elif encoding=='he':
            enc=ce.HashingEncoder()
        elif encoding=='oe':
            enc=ce.OrdinalEncoder()
        elif encoding=='ohe':
            enc=ce.BaseNEncoder(base=1)
        enc.fit(train_df)
        train_enc=enc.transform(train_df)
        valid_enc=enc.transform(valid_df)
        print('category encoding completed', '\n')
        categorical_names = {}
        return train_enc, valid_enc, categorical_names
    
    

    
""" 
below class was taken from url=https://www.kaggle.com/superant/oh-my-cat
Thermometer encoding (believed to be working really good for GANs)
cannot handle unseen values in test. so use for situations where all levels for a cat variable has atleast 1 sample in train
"""

from sklearn.base import TransformerMixin
from itertools import repeat
import scipy


class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result