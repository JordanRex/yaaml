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
