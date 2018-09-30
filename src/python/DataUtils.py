import pandas as pd


class DataUtils:

    categorical_types = ['object']

    @staticmethod
    def get_categorical_data(data):
        return data.select_dtypes(DataUtils.categorical_types)

    @staticmethod
    def get_numerical_data(data):
        return data.select_dtypes(exclude=DataUtils.categorical_types)

    @staticmethod
    def concat_dfs_ignoring_index(df1, df2):
        if df1 is None:
            return df2
        if df2 is None:
            return df1
        df1_no_index = df1.reset_index(drop=True, inplace=False)
        df2_no_index = df2.reset_index(drop=True, inplace=False)
        concated_data = pd.concat([df1_no_index, df2_no_index], axis=1)
        return concated_data
