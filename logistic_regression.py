import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

def logistic_regression(df):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    return lr, X_test, y_test, y_pred


def process_logistic_regression(df, data_name):
    lr, X_test, y_test, y_pred = logistic_regression(df)
    print("Accuracy: ", lr.score(X_test, y_test))

    # Merge X_test, y_test, y_pred into one dataframe
    df_result = pd.DataFrame(X_test)
    df_result.insert(0, 'Peptide', y_test)
    df_result.insert(1, 'Peptide_pred', y_pred)
    df_result.rename_axis("id", axis=1, inplace=True)
    df_result.rename(columns={'index': 'Peptide'}, inplace=True)

    # Save to file
    df_result.to_csv('data/logistic_regression-' + data_name + '.csv')
    df_result.to_pickle('data/logistic_regression-' + data_name + '.pkl')
    return lr, df_result

def hypothesis_test(df, top_features):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]

    for feature in top_features:
        # Get independent variable
        top_feature_name = feature[0]
        X_new = X[top_feature_name]
        X_new = sm.add_constant(X_new)

        # Fit into model
        model = sm.Logit(y, X_new)
        result = model.fit()
        
        print(result.summary())


    return 

def mann_whitney_test(df, top_features):
    # extract the top feature names
    top_feature_names = [feature[0] for feature in top_features]

    # separate your data based on the target variable
    group1 = df[df["Peptide"] == 0]
    group2 = df[df["Peptide"] == 1]

    for feature in top_feature_names:
        # perform Mann-Whitney U test
        stat, p = mannwhitneyu(group1[feature], group2[feature])
        print('Features: %s, Statistics=%.3f, p=%.3f' % (feature, stat, p))

    return


if __name__ == '__main__':
    # data_set = [
    #     "variants_processed_knn",
    #     "variants_processed_mean",
    #     "knn_PCA",
    #     "mean_PCA",
    #     "selected_features_knn",
    #     "selected_features_mean"
    # ]
    data_set = [
        "selected_features_knn",
    ]

    for data_name in data_set:
        print("=====================================")
        print("Data set:", data_name)
        df = pd.read_pickle('data/' + data_name + '.pkl')
        lr, df_result = process_logistic_regression(df, data_name)

        # show the top 10 features by weight
        # sort the weight by absolute value descending order
        df_weight = pd.DataFrame(lr.coef_[0], index=df.drop("Peptide", axis=1).columns, columns=['weight'])
        df_weight['abs_weight'] = df_weight['weight'].abs()
        df_weight.sort_values(by=['abs_weight'], ascending=False, inplace=True)
        # print(df_weight.head(10))
        print("Top 10 features name and weight by weight")
        for index, row in df_weight.head(10).iterrows():
            print(index, row['weight'])

        # extract the top features for hypothesis testing
        top_features = [(index, row['weight']) for index, row in df_weight.head(10).iterrows()]

        # perform hypothesis tests
        hypothesis_test(df, top_features)

        # perform mann whitney tests
        mann_whitney_test(df, top_features)

