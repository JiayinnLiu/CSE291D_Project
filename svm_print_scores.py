import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from helper import cross_valid
from helper import get_evaluation_scores

def svm(df):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
    X_train_all, X_test_all, y_train_all, y_test_all = cross_valid(5, X, y)
    f1s = []
    precisions = []
    recalls = []
    for i in range(5):
        X_train, X_test, y_train, y_test = X_train_all[i], X_test_all[i], y_train_all[i], y_test_all[i]

        lr = SVC()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        f1, prec, rec = get_evaluation_scores(y_test, y_pred)
        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
    return np.mean(f1s), np.mean(precisions), np.mean(recalls)


def process_svm(df):
    f1_score, precision, recall = svm(df)
    print("f1 score: ", f1_score)
    print("precision: ", precision)
    print("recall: ", recall)
    print()


if __name__ == '__main__':
    data_set = [
        "variants_processed_knn",
        "variants_processed_mean",
        "knn_PCA",
        "mean_PCA",
        "selected_features_knn",
        "selected_features_mean"
    ]

    for data_name in data_set:
        print("Data set:", data_name)
        df = pd.read_pickle('Data/' + data_name + '.pkl')
        #lr, df_result = process_random_forest(df, data_name)
        process_svm(df)