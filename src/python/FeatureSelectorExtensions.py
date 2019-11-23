import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from DataUtils import DataUtils
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier


class CategoricalValueCountThresholdRemover(BaseEstimator, TransformerMixin):
    def __init__(self,
                 max_categories_in_single_variable=5,
                 min_categories_in_single_variable=2):
        # Params
        self.max_categories_in_single_variable = \
            max_categories_in_single_variable
        self.min_categories_in_single_variable = \
            min_categories_in_single_variable
        #
        self.categorical_variables_to_remove = None

    def fit(self, X, y=None):
        categorical_data = DataUtils.get_categorical_data(X)
        # Drop categorical with a lot of categories
        cat_sizes = pd.Series(
            [categorical_data[c].value_counts().size for c
             in categorical_data],
            index=categorical_data.columns)
        sparse_categories = \
            cat_sizes.loc[cat_sizes >
                          self.max_categories_in_single_variable]
        skewed_categories = \
            cat_sizes.loc[cat_sizes <
                          self.min_categories_in_single_variable]
        self.categorical_variables_to_remove = sparse_categories.append(
            skewed_categories)

        # Return fit object
        return self

    def transform(self, X):
        categorical_data = DataUtils.get_categorical_data(X)
        return categorical_data.drop(
            list(self.categorical_variables_to_remove.index), axis=1)


class SafeNamedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None, excluded_features=None):
        if(selected_features is None and excluded_features is None):
            raise ValueError(
                "At least one of selected/excluded_features needs to be full!")
        if(selected_features is not None and excluded_features is not None):
            raise ValueError(
                "Only one of selected/excluded_features can be full!")
        self.selected_features = selected_features
        self.excluded_features = excluded_features
        self.columns_to_drop = None

    def fit(self, X):
        if self.selected_features is not None:
            self.columns_to_drop = [
                c for c in X.columns if c not in self.selected_features]
        elif self.excluded_features is not None:
            self.columns_to_drop = [
                c for c in X.columns if c in self.excluded_features]
        return self

    def transform(self, X):
        if self.columns_to_drop is not None:
            return X.drop(self.columns_to_drop, axis=1, errors='ignore')
        else:
            return X


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, min_variance_allowed=0.01):
        self.min_variance_allowed = min_variance_allowed
        self.variance_selector = VarianceThreshold(
            threshold=min_variance_allowed)

    def fit(self, X, y=None):
        self.variance_selector.fit(X)
        # Return fit object
        return self

    def transform(self, X):
        return X[X.columns[self.variance_selector.get_support(indices=True)]]


class ExtraTreesImportanceFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importance_threshold=0.03, seed=666):
        self.seed = seed
        self.feature_importance_threshold = feature_importance_threshold
        self.feature_importances = None
        self.selected_features = None

    def fit(self, X, y):
        all_feature_names = X.columns.values
        # Train model and get importance metrics
        feature_selection_model = ExtraTreesClassifier(
            n_estimators=50, n_jobs=-1, random_state=self.seed).fit(X, y)
        self.feature_importances = feature_selection_model.feature_importances_
        selector = SelectFromModel(
            feature_selection_model,
            prefit=True, threshold=self.feature_importance_threshold)
        X_new = selector.transform(X)
        remaining_feature_indices = sorted(
            range(len(self.feature_importances)),
            key=lambda i: self.feature_importances[i])[-X_new.shape[1]:]
        self.selected_features = np.array(all_feature_names)[
            remaining_feature_indices]
        # Return fit object
        return self

    def transform(self, X):
        return X.loc[:, self.selected_features]


class Chi2KSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=6):
        self.k = k
        self.chi2k_selector = SelectKBest(chi2, k=self.k)

    def fit(self, X, y):
        self.chi2k_selector.fit(X, y)
        # Return fit object
        return self

    def transform(self, X):
        return self.chi2k_selector.transform(X)


class LassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, C=0.005, seed=666):
        self.seed = seed
        self.C = C
        self.lasso_selector = SelectFromModel(
            LinearSVC(C=self.C, penalty="l1",
                      dual=False, random_state=self.seed))

    def fit(self, X, y):
        if self.C is not None:
            self.lasso_selector.fit(X, y)
            self.selected_features = (
                X[X.columns[self.lasso_selector.get_support(
                    indices=True)]]).columns.values
        # Return fit object
        return self

    def transform(self, X):
        if self.C is not None:
            return X[X.columns[self.lasso_selector.get_support(indices=True)]]
        else:
            return X
