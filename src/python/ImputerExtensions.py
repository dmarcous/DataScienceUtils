import pandas as pd
import numpy as np
from DataUtils import DataUtils
from sklearn.preprocessing import Imputer


class NumericConstantImputer(BaseEstimator, TransformerMixin):
    def __init__(self, constant):
        self.constant = constant

    def fit(self, X, y=None):
        X.fillna(self.constant)

        # Return fit object
        return self

    def transform(self, X):
        return X.fillna(self.constant)


class SafeNumericImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_imputation_mode='median'):
        self.numeric_imputation_mode = numeric_imputation_mode
        self.numeric_imputer = Imputer(strategy=self.numeric_imputation_mode)
        self.successfully_imputed_columns = None

    def fit(self, X):
        numerical_data = DataUtils.get_numerical_data(X)
        full_numerical_data = self.numeric_imputer.fit_transform(
            numerical_data)
        self.successfully_imputed_columns = numerical_data.columns[~np.isnan(
            self.numeric_imputer.statistics_)]
        # Return fit object
        return self

    def transform(self, X):
        numerical_data = DataUtils.get_numerical_data(X)
        full_numerical_data = self.numeric_imputer.transform(numerical_data)
        # Safe remove columns which are all empty
        successfully_imputed_columns = numerical_data.columns[~np.isnan(
            self.numeric_imputer.statistics_)]
        full_numerical_data = pd.DataFrame(
            full_numerical_data, columns=successfully_imputed_columns)
        # Drop columns not in fit self.successfully_imputed_columns
        columns_to_drop = [
            c for c in full_numerical_data.columns if
            c not in self.successfully_imputed_columns]
        if len(columns_to_drop) > 0:
            full_numerical_data = full_numerical_data.drop(
                columns_to_drop, axis=1, errors='ignore')
        # Add columns in self.successfully_imputed_columns and not in current
        columns_to_add = [
            c for c in self.successfully_imputed_columns if
            c not in full_numerical_data.columns]
        if len(columns_to_add) > 0:
            full_numerical_data = full_numerical_data.reindex(
                columns=[*full_numerical_data.columns.tolist(), *columns_to_add],
                fill_value=0)

        return full_numerical_data


class CategoricalModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.frequent_vals = None

    def fit(self, X):
        categorical_data = DataUtils.get_categorical_data(X)
        # Categorical mode imputer
        self.frequent_vals = pd.Series(
            [categorical_data[c].value_counts().index[0]
             for c in categorical_data],
            index=categorical_data.columns)

        # Return fit object
        return self

    def transform(self, X):
        categorical_data = DataUtils.get_categorical_data(X)
        return categorical_data.fillna(self.frequent_vals)


class CategoricalNewCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, null_value="Null_Cat"):
        self.null_value = null_value

    def fit(self, X):
        # Return fit object
        return self

    def transform(self, X):
        categorical_data = DataUtils.get_categorical_data(X)
        return categorical_data.fillna(self.null_value)
