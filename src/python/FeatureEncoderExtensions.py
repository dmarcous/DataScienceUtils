import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from DataUtils import DataUtils


class SafeMultiColumnCategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, unseen_categories_fill_value=0):
        self.unseen_categories_fill_value = unseen_categories_fill_value
        self.encoded_columns = None
        self.encoding_map = {}

    def fit(self, X, y=None):
        categorical_data = DataUtils.get_categorical_data(X)
        # Encode to dummies
        encoded_data = pd.get_dummies(categorical_data)
        self.encoded_columns = encoded_data.columns

        self.encoding_map = {orig_col:
                             [col for col in self.encoded_columns.values
                              if col.startswith(
                                  orig_col)]
                             for orig_col in categorical_data.columns.values}
        self.decoding_map = {v: k for k,
                             vl in self.encoding_map.items() for v in vl}

        # Return fit object
        return self

    def transform(self, X):
        categorical_data = DataUtils.get_categorical_data(X)
        # Encode to dummies
        encoded_data = pd.get_dummies(categorical_data)
        # Create final encoded df
        return encoded_data.reindex(
            columns=self.encoded_columns,
            fill_value=self.unseen_categories_fill_value)
