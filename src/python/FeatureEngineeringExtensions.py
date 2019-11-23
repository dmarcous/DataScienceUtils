import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype

# Example usage :
# feature_engineer = ExpressionBasedFeatureEngineering({
#   "x_lte_90": "x <= 90",
#   "x_comp_cond": "x_lte_90 == True and x <= 80",
# })


class ExpressionBasedFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, engineered_expressions=None):
        # Params
        self.engineered_expressions = engineered_expressions
        #
        self.base_features = []
        self.final_features = []

    def __withColumn(df, feature_name, engineering_expression):
        index_resolvers = df._get_index_resolvers()
        resolvers = dict(df.iteritems()), index_resolvers
        df[feature_name] = pd.eval(engineering_expression, resolvers=resolvers)
        return df

    def fit(self, X, y=None):
        if self.engineered_expressions is not None:
            eng_X = X.copy()
            for k, v in self.engineered_expressions.items():
                base_candidates = [cd for cd in v.split(
                ) if cd in X.columns.values and cd not in self.base_features]
                eng_X = self.__withColumn(eng_X, k, v)
                self.final_features.append(k)
                self.base_features.extend(base_candidates)

        # Return fit object
        return self

    def transform(self, X):
        eng_X = X.copy()
        if self.engineered_expressions is not None:
            for k, v in self.engineered_expressions.items():
                eng_X = self.__withColumn(eng_X, k, v)

        return eng_X


# Example usage :
# cross_engineered_features =
#     CrossFeatureEngineering(
#         feature_groups=[["x_lte_80", "x", "y"],
#                         ["x_lte_80", "x", "z"]])
class CrossFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, feature_groups=None):
        # Params
        self.feature_groups = feature_groups
        #
        self.base_features = []
        self.final_features = []

        def __crossFeaturePair(self, df, feature1, feature2):
            if is_numeric_dtype(df[feature1]) and \
                    is_numeric_dtype(df[feature2]):
                crossed_name = feature1 + "_mult_" + feature2
                df[crossed_name] = df[feature1] * df[feature2]
            elif not is_numeric_dtype(df[feature1]) or \
                    not is_numeric_dtype(df[feature2]):
                crossed_name = feature1 + "_concat_" + feature2
                df[crossed_name] = df[feature1].map(
                    str) + "_" + df[feature2].map(str)
            else:
                raise ValueError("cant get here")

            return df

        def __crossFeatures(self, df, feature_names):
            engineered_data = df.copy()
            num_features = len(feature_names)
            for featureA_index, featureA in \
                    enumerate(feature_names[:num_features - 1]):
                for featureB_index, featureB in \
                        enumerate(feature_names[featureA_index + 1:]):
                    engineered_data = self.__crossFeaturePair(
                        engineered_data, featureA, featureB)

            return engineered_data

        def __crossFeatureGroups(self, df, feature_groups):
            engineered_data = df.copy()
            for feature_group in feature_groups:
                engineered_data = self.__crossFeatures(
                    engineered_data, feature_group)

            return engineered_data

    def fit(self, X, y=None):
        if self.feature_groups is not None:
            base_candidates = list(
                set([col for clist in self.feature_groups for col in clist]))
            self.base_features.extend(
                [col for col in base_candidates
                 if col not in self.base_features])
            eng_X = X.copy()
            eng_X = self.__crossFeatureGroups(eng_X, self.feature_groups)
            self.final_features = [
                col for col in eng_X.columns.values
                if col not in X.columns.values]

        # Return fit object
        return self

    def transform(self, X):
        eng_X = X.copy()
        if self.feature_groups is not None:
            eng_X = self.__crossFeatureGroups(eng_X, self.feature_groups)

        return eng_X
