# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def decision_tree(df):
    dfc = df.copy()
    X, y = dfc.drop("Peptide", axis=1), df["Peptide"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    return dt, X_test, y_test, y_pred


def process_decision_tree(df, data_name):
    dt, X_test, y_test, y_pred = decision_tree(df)
    print("Accuracy: ", dt.score(X_test, y_test))

    # Merge X_test, y_test, y_pred into one dataframe
    df_result = pd.DataFrame(X_test)
    df_result.insert(0, 'Peptide', y_test)
    df_result.insert(1, 'Peptide_pred', y_pred)
    df_result.rename_axis("id", axis=1, inplace=True)
    df_result.rename(columns={'index': 'Peptide'}, inplace=True)

    # Save to file
    df_result.to_csv('data/decisionTree-' + data_name + '.csv')
    df_result.to_pickle('data/decisionTree-' + data_name + '.pkl')
    return dt, df_result

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
        lr, df_result = process_decision_tree(df, data_name)