import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from utils import *

def GetDummies(data_set, categorical_features):
    """ Reserve the origin attribute while getting dummies """
    reserve_name = data_set.name
    reserve_trn_len = data_set.trn_len
    data_set = pd.get_dummies(data_set, columns=categorical_features, drop_first=True)
    data_set.name = reserve_name
    data_set.trn_len = reserve_trn_len
    return data_set

def FeatureEngineering(df_data_set: pd.DataFrame):
    """ As its name suggests, do feature engineering """
    # Get numerical and categorical features
    numerical_features = df_data_set.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df_data_set.select_dtypes(exclude=["int64", "float64"]).columns
    # For all the numerical features, do min-max normalization
    # For all the categorical features, do dummy coding
    df_data_set[numerical_features] = MinMaxScaler().fit_transform(df_data_set[numerical_features])
    df_data_set = GetDummies(df_data_set, categorical_features)

    return DivideDF(df_data_set)

if __name__ == "__main__":
    # Get data set
    df_train_set, df_test_set = GetDataSet("./data")

    # # Save information to markdown files
    # Save2Markdown(df_train_set, "./analysis")
    # Save2Markdown(df_test_set, "./analysis")

    # Deal with missing values
    DealWithMissingValues(df_train_set)
    DealWithMissingValues(df_test_set)

    # Create df_all to create format features
    train_y = df_train_set["SalePrice"]
    df_train_set.drop("SalePrice", axis=1, inplace=True)
    df_all = ConcatDF(df_train_set, df_test_set)
    df_all.name = "all"

    # Feature engineering
    # Honestly speaking, there is no need to return y for FE
    train_X, test_X = FeatureEngineering(df_all)

    # # Detect missing values
    # # for column in df_train_set.columns:
    # for column in df_train_set.columns:
    #     missing_line_count = df_train_set[column].isnull().sum()
    #     if missing_line_count != 0:
    #         print(column)
    #         print(df_train_set[column][df_train_set[column].isnull().values])

    # Machine learning
    # There is no train_test split, because we can get score directly form lb
    lr = LinearRegression().fit(train_X, train_y)
    result = lr.predict(test_X)

    # Output
    output = pd.DataFrame({"Id": range(1461, 2920), "SalePrice": result})
    output.to_csv("./result.csv", index=False)
