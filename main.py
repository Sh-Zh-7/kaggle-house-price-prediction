from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
import xgboost

from utils import *
import warnings
warnings.filterwarnings("ignore")

# Constant
SEED = 233

class StackingAverageModel(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, base_models, meta_model, n_fold=5):
        self.base_models = base_models
        # The clone here is only for pretending data leakage
        self.base_models_clone = [[clone(base_model)] * n_fold for base_model in base_models]
        self.meta_model = meta_model
        self.n_fold = n_fold

    def fit(self, X, y):
        """ Fit the X and y by using out-of-fold prediction """
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=SEED)
        out_of_fold_prediction = np.zeros((X.shape[0], len(self.base_models)))
        for i, base_model in enumerate(self.base_models):
            for j, (train_index, test_index) in enumerate(skf.split(X, y)):
                model = self.base_models_clone[i][j]
                model.fit(X[train_index], y[train_index])
                out_of_fold_prediction[test_index, i] = model.predict(X[test_index])
        self.meta_model.fit(out_of_fold_prediction, y)

    def predict(self, X):
        """ Make predictions on the whole base models """
        meta_features = np.column_stack([
            np.column_stack([base_model.predict(X) for base_model in base_models]).mean(axis=1)
            for base_models in self.base_models_clone]
        )
        return self.meta_model.predict(meta_features)

# Press Ctrl+- to hide the detail of the functions
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

def LoadModels(path):
    """ Load base models and meta model for model ensemble """
    # Get all base models
    base_models = []
    base_model_dir = os.path.join(path, "base_models")
    base_models_name = ["elastic_net", "gradient_boosting", "kernel_ridge", "lasso"]
    base_models_cntr = [ElasticNet, GradientBoostingRegressor, KernelRidge, Lasso]
    for base_model, cntr in zip(base_models_name, base_models_cntr):
        base_model_file = base_model + ".json"
        base_model = cntr(**Params(os.path.join(base_model_dir, base_model_file)).dict)
        base_models.append(base_model)
    # Get the meta model
    meta_model_path = os.path.join(path, "meta_model/xgboost.json")
    meta_model = xgboost.XGBRegressor(**Params(meta_model_path).dict)

    return base_models, meta_model


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
    train_y = df_train_set["SalePrice"].values
    df_train_set.drop("SalePrice", axis=1, inplace=True)
    df_all = ConcatDF(df_train_set, df_test_set)
    df_all.name = "all"

    # Feature engineering
    # Honestly speaking, there is no need to return y for FE
    # Remember to convert it to numpy object
    train_X, test_X = FeatureEngineering(df_all)
    train_X = train_X.values
    test_X = test_X.values

    # # Detect missing values
    # # for column in df_train_set.columns:
    # for column in df_train_set.columns:
    #     missing_line_count = df_train_set[column].isnull().sum()
    #     if missing_line_count != 0:
    #         print(column)
    #         print(df_train_set[column][df_train_set[column].isnull().values])

    # Machine learning
    # There is no train_test split, because we can get score directly form lb
    base_models, meta_model = LoadModels("./models")
    stacking_model = StackingAverageModel(base_models, meta_model)
    stacking_model.fit(train_X, train_y)
    result = stacking_model.predict(test_X)

    # Output
    output = pd.DataFrame({"Id": range(1461, 2920), "SalePrice": result})
    output.to_csv("./result.csv", index=False)
