import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def support_vector_machine(df):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    svm = SVC()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    return svm, X_test, y_test, y_pred


def process_svm(df, data_name):
    svm, X_test, y_test, y_pred = support_vector_machine(df)
    print("Accuracy: ", svm.score(X_test, y_test))

    # Merge X_test, y_test, y_pred into one dataframe
    df_result = pd.DataFrame(X_test)
    df_result.insert(0, 'Peptide', y_test)
    df_result.insert(1, 'Peptide_pred', y_pred)
    df_result.rename_axis("id", axis=1, inplace=True)
    df_result.rename(columns={'index': 'Peptide'}, inplace=True)

    # Save to file
    df_result.to_csv('data/svm-' + data_name + '.csv')
    df_result.to_pickle('data/svm-' + data_name + '.pkl')
    return svm, df_result



if __name__ == '__main__':
    data_set = [
        "variants_processed_knn",
        "variants_processed_mean",
        "knn_PCA",
        "mean_PCA",
    ]

    for data_name in data_set:
        print("Data set:", data_name)
        df = pd.read_pickle('data/' + data_name + '.pkl')
        lr, df_result = process_svm(df, data_name)