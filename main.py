from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import xgboost

from utils import *
from feat_eng import FeatureEngineering
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
        # We always use stratified-k-fold in classification tasks
        # In regression tasks, we just simply use k-fold
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=SEED)
        out_of_fold_prediction = np.zeros((X.shape[0], len(self.base_models)))
        for i, base_model in enumerate(self.base_models):
            for j, (train_index, test_index) in enumerate(kf.split(X, y)):
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

def LoadModels(path):
    """ Load base models and meta model for model ensemble """
    # Get base models that need hyper-parameters
    base_models = []
    base_model_dir = os.path.join(path, "base_models")
    base_models_name = ["elastic_net", "gradient_boosting", "kernel_ridge", "lasso"]
    base_models_cntr = [ElasticNet, GradientBoostingRegressor, KernelRidge, Lasso]
    for base_model, cntr in zip(base_models_name, base_models_cntr):
        base_model_file = base_model + ".json"
        base_model = cntr(**Params(os.path.join(base_model_dir, base_model_file)).dict)
        base_models.append(base_model)
    # Get base models without hyper-parameters
    lr = make_pipeline(RobustScaler(), LinearRegression())
    base_models.append(lr)
    # Get the meta model
    meta_model_path = os.path.join(path, "meta_model/xgboost.json")
    meta_model = xgboost.XGBRegressor(**Params(meta_model_path).dict)

    return base_models, meta_model

if __name__ == "__main__":
    # Get test set, train set and its target values
    df_train_set, df_test_set = GetDataSet("./data")
    train_y = df_train_set["SalePrice"].values
    df_train_set.drop("SalePrice", axis=1, inplace=True)

    # Deal with missing values
    DealWithMissingValues(df_train_set)
    DealWithMissingValues(df_test_set)

    # Feature engineering
    # Honestly speaking, there is no need to return y for FE
    # Remember to convert it to numpy object
    train_X, test_X = FeatureEngineering(df_train_set, df_test_set)

    # Target engineering
    # Don't forget to convert to original at the end
    train_y = np.log1p(train_y)

    # Machine learning
    # There is no train_test split, because we can get score directly form lb
    base_models, meta_model = LoadModels("./models")
    stacking_model = StackingAverageModel(base_models, meta_model)
    stacking_model.fit(train_X, train_y)
    result = np.expm1(stacking_model.predict(test_X))

    # Output
    output = pd.DataFrame({"Id": range(1461, 2920), "SalePrice": result})
    output.to_csv("./result.csv", index=False)
