from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.preprocessing import StandardScaler

from helper import *

def FeatureEngineering(df_train_set, df_test_set):
    """ As its name suggest, do feature engineering """
    # Deal with train set and test set separately
    FeatureEngineeringSeparately(df_train_set)
    FeatureEngineeringSeparately(df_test_set)
    # Concat the train and test set to solve some annoying problems
    df_all = ConcatDF(df_train_set, df_test_set)
    df_all.name = "all"
    df_train_set, df_test_set = FeatureEngineeringAll(df_all)
    # Convert to numpy object
    return df_train_set.values, df_test_set.values

def FeatureEngineeringSeparately(df_data_set):
    # Check
    numerical_features = df_data_set.select_dtypes(include=["int64", "float64"]).columns
    # Check the skew of all numerical features
    skewed_feats = df_data_set[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    # Get high skewness features
    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    # lambda is a hyper-parameter to tune
    lam = 0.15
    for feat in skewed_features:
        df_data_set[feat] = boxcox1p(df_data_set[feat], lam)

def FeatureEngineeringAll(df_data_set):
    """ As its name suggests, do feature engineering """
    # Get numerical and categorical features
    numerical_features = df_data_set.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df_data_set.select_dtypes(exclude=["int64", "float64"]).columns
    # For all the numerical features, do min-max normalization
    # For all the categorical features, do dummy coding
    df_data_set[numerical_features] = StandardScaler().fit_transform(df_data_set[numerical_features])
    df_data_set = GetDummies(df_data_set, categorical_features)

    return DivideDF(df_data_set)
