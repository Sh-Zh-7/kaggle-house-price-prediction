from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from utils import *

sns.set()

if __name__ == "__main__":
    # Get train set and test set
    # Notice the pwd is still the project repo
    df_train_set, df_test_set = GetDataSet("./data")

    # # Save information to markdown files
    # Save2Markdown(df_train_set, "./analysis")
    # Save2Markdown(df_test_set, "./analysis")

    # Plot
    # Target engineering
    df_train_set["SalePrice"] = np.log1p(df_train_set["SalePrice"])
    # The distribution of target variable
    sns.distplot(df_train_set["SalePrice"], fit=norm)
    plt.savefig("./analysis/target_dist_plot.png")
    # The PP plot of target variable
    stats.probplot(df_train_set["SalePrice"], plot=plt)
    plt.savefig("./analysis/target_PP_plot.png")

    # Check
    numerical_features = df_train_set.select_dtypes(include=["int64", "float64"]).columns
    # Check the skew of all numerical features
    skewed_feats = df_train_set[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    # Get high skewness features
    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    # Lambda is a hyper-parameter to tune
    lam = 0.15
    for feat in skewed_features:
        df_train_set[feat] = boxcox1p(df_train_set[feat], lam)


