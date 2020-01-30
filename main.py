import numpy as np

from utils.helper import *
import utils.model as model
from utils.feat_eng import FeatureEngineering

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Get test set, train set and its target values
    print("Reading..")
    df_train_set, df_test_set = GetDataSet("./data")
    train_y = df_train_set["SalePrice"].values
    df_train_set.drop("SalePrice", axis=1, inplace=True)
    print("Done.")

    # Deal with missing values
    print("Pre-processing..")
    DealWithMissingValues(df_train_set)
    DealWithMissingValues(df_test_set)
    # Feature engineering
    # Honestly speaking, there is no need to return y for FE
    # Remember to convert it to numpy object
    train_X, test_X = FeatureEngineering(df_train_set, df_test_set)
    # Target engineering
    # Don't forget to convert to original at the end
    train_y = np.log1p(train_y)
    print("Done.")

    # Machine learning
    # There is no train_test split, because we can get score directly form lb
    print("Training..")
    base_models, meta_model = model.LoadModels("./models")
    stacking_model = model.StackingAverageModel(base_models, meta_model)
    stacking_model.fit(train_X, train_y)
    result = np.expm1(stacking_model.predict(test_X))
    print("Done.")

    # Save parameters and throw them to the back-end
    stacking_model.save_params("./models")

    # # Output
    print("Writing to csv..")
    # output = pd.DataFrame({"Id": range(1461, 2920), "SalePrice": result})
    # output.to_csv("./result.csv", index=False)
    print("Done!")
