from utils.helper import *
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import StandardScaler, LabelEncoder

def FeatureEngineering(df_train_set, df_test_set):
    """ As its name suggests, do feature engineering """
    # Deal with train set and test set separately
    FeatureEngineeringSeparately(df_train_set)
    FeatureEngineeringSeparately(df_test_set)
    # Concat the train and test set to solve some annoying problems
    df_all = ConcatDF(df_train_set, df_test_set)
    df_all.name = "all"
    df_train_set, df_test_set = FeatureEngineeringAll(df_all)
    # Convert to numpy object
    return df_train_set.values, df_test_set.values

# ---------------------------------------Separately---------------------------------------------------------------
def FeatureEngineeringSeparately(df_data_set):
    """ Do feature engineering on separate data set. """
    DealWithSkewedFeatures(df_data_set)
    # TODO: Add more feature engineering tricks

def DealWithSkewedFeatures(df_data_set):
    """ Select high skewness features and remove all of them. """
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

# ----------------------------------------All-------------------------------------------------------------------
def FeatureEngineeringAll(df_data_set):
    """ Do feature engineering on all of the data set. """
    # Get numerical and categorical features
    numerical_features = df_data_set.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df_data_set.select_dtypes(exclude=["int64", "float64"]).columns
    # Separate label encoder features and dummy code features
    # label_encode_features = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    #     'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
    #     'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
    #     'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
    #     'YrSold', 'MoSold']
    # dummy_code_features = [dummy_code_feature for dummy_code_feature in categorical_features
    #                        if dummy_code_feature not in label_encode_features]
    # Dealing
    df_data_set[numerical_features] = StandardScaler().fit_transform(df_data_set[numerical_features])
    # for label_encode_feature in label_encode_features:
    #     df_data_set[label_encode_feature] = LabelEncoder().fit_transform(list(df_data_set[label_encode_feature]))
    df_data_set = GetDummies(df_data_set, categorical_features)

    return DivideDF(df_data_set)
