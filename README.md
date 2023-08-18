# CSE 291

A brief description

## Data Preprocessing

In this project, we perform the following data preprocessing steps to clean and prepare the dataset for further analysis:

### 1. Removing NaN columns

#### 1.1 Removing only columns all is NaN
We removed any columns with NaN values, ensuring that the dataset only contains columns with complete data. After removing the dataset is $92 \times 100941$.

#### 1.2 Removing columns over 80% rows are NaN
We removed columns with over 80% rows are NaN, to further ensure the dataset has more valueable data. After removing the dataset is $92 \times 59589$.

### 2. Filling NaN values

We explored two different methods for filling missing values in the dataset:

#### 2.1 Mean Imputation

For this method, we calculated the mean value of each column and replaced the missing values (NaN) with the corresponding column mean.

```python
# Calculate column means
column_means = data.mean()

# Fill NaN values with column means
data.fillna(column_means, inplace=True)
```

#### 2.2 K-Nearest Neighbors (KNN) Imputation

In this method, we used the KNN algorithm to fill missing values in the dataset. The KNN imputation method estimates missing values based on the similarity of instances with complete data.

```python 
from sklearn.impute import KNNImputer

# Instantiate KNN imputer
imputer = KNNImputer(n_neighbors=3)

# Impute missing values using KNN
data_imputed = imputer.fit_transform(data)

# Convert imputed data back to DataFrame
data = pd.DataFrame(data_imputed, columns=data.columns)
```

Zhaoxing Lyu May 8th

Use mean to find the top 10 most important features after selecting by a Random Forest model

Meet Singular matrix issue when doing hypothesis_test

Perform t-test successfully by using scipy.stats 