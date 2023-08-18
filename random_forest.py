import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from helper import cross_valid
from helper import get_evaluation_scores
import statsmodels.api as sm
from scipy.stats import ttest_ind

# def random_forest(df, random_state=None):
#     dfc = df.copy()
#     X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
#     X_train_all, X_test_all, y_train_all, y_test_all = cross_valid(5, X, y)
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     f1s = []
#     precisions = []
#     recalls = []
#     for i in range(5):
#         X_train, X_test, y_train, y_test = X_train_all[i], X_test_all[i], y_train_all[i], y_test_all[i]

#         lr = RandomForestClassifier(random_state=random_state)
#         lr.fit(X_train, y_train)

#         y_pred = lr.predict(X_test)
#         f1, prec, rec = get_evaluation_scores(y_test, y_pred)
#         f1s.append(f1)
#         precisions.append(prec)
#         recalls.append(rec)
#     return np.mean(f1s), np.mean(precisions), np.mean(recalls)
    #return lr, X_test, y_test, y_pred

# return a list of top features, sorted by their importance, in addition to the F1 score, precision, and recall. 
# the feature can be further analyzed 
def random_forest(df, random_state=None):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
    X_train_all, X_test_all, y_train_all, y_test_all = cross_valid(5, X, y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    f1s = []
    precisions = []
    recalls = []
    feature_importances = []

    for i in range(5):
        X_train, X_test, y_train, y_test = X_train_all[i], X_test_all[i], y_train_all[i], y_test_all[i]

        lr = RandomForestClassifier(random_state=random_state)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        f1, prec, rec = get_evaluation_scores(y_test, y_pred)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        # add the feature importance to the list
        feature_importances.append(lr.feature_importances_)
    
    # Get the mean importance for each feature across all models
    mean_feature_importances = np.mean(feature_importances, axis=0)
    
    # Combine feature names and their importance scores
    feature_importance_dict = dict(zip(X.columns, mean_feature_importances))
    
    # Sort features by their importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Return the top features
    top_features = sorted_features[:10]  # change the number here to get more or less features

    return np.mean(f1s), np.mean(precisions), np.mean(recalls), top_features

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

    # # Extract the top feature names
    # top_feature_names = [feature[0] for feature in top_features]
    # print(top_feature_names)

    # # Extract these features from your dataframe
    # X = X[top_feature_names]

    # # Add constant to the features
    # X = sm.add_constant(X)

    # # Fit the Logistic Regression Model
    # model = sm.Logit(y, X)
    # result = model.fit()

    # # Print the summary
    # print(result.summary())

    # return result

def t_test(df, top_features):
    # extract the top feature names
    top_feature_names = [feature[0] for feature in top_features]

    # separate data based on the target variable
    group1 = df[df["Peptide"] == 0]
    group2 = df[df["Peptide"] == 1]

    # print(group1)
    # print(group2)

    for feature in top_feature_names:
        stat, p = ttest_ind(group1[feature], group2[feature])
        print('Features: %s, Statistics=%.3f, p=%.3f' % (feature, stat, p))

    return

def process_random_forest(df, data_name):
    f1_score, precision, recall, top_features = random_forest(df)
    print("f1 score: ", f1_score)
    print("precision: ", precision)
    print("recall: ", recall)
    print("top_features: ", top_features)
    print()
    hypothesis_test(df, top_features)
    t_test(df, top_features)

    # lr, X_test, y_test, y_pred = random_forest(df)
    # print("Accuracy: ", lr.score(X_test, y_test))

    # # Merge X_test, y_test, y_pred into one dataframe
    # df_result = pd.DataFrame(X_test)
    # df_result.insert(0, 'Peptide', y_test)
    # df_result.insert(1, 'Peptide_pred', y_pred)
    # df_result.rename_axis("id", axis=1, inplace=True)
    # df_result.rename(columns={'index': 'Peptide'}, inplace=True)

    # # Save to file
    # newpath = r'Data/random_forest'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # df_result.to_csv('Data/random_forest/random_forest-' + data_name + '.csv')
    # df_result.to_pickle('Data/random_forest/random_forest-' + data_name + '.pkl')
    # return lr, df_result


if __name__ == '__main__':
    data_set = [
        # "variants_processed_knn",
        # "variants_processed_mean",
        # "knn_PCA",
        # "mean_PCA",
        "selected_features_knn",
        # "selected_features_mean"
    ]

    for data_name in data_set:
        print("=====================================")
        print("Data set:", data_name)
        df = pd.read_pickle('Data/' + data_name + '.pkl')
        #lr, df_result = process_random_forest(df, data_name)
        process_random_forest(df, data_name)