import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report

def data_preprocessing(path_to_data):

    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]

    # Load the dataset
    df = pd.read_csv(path_to_data, 
                    names=columns, 
                    skipinitialspace=True,
                    skiprows=1 if "test" in path_to_data else 0,)
    # Replace empty values with NaN
    df.replace("?", np.nan, inplace=True)
    #Drop NaN entries
    # df.dropna(how="any", inplace=True)
    #Remove duplicates
    df.drop_duplicates(inplace=True)

    # Drop unnecessary columns
    df.drop(["fnlwgt","education","relationship", "race"], axis=1, inplace=True)
    # Combine capital-gain and capital-loss into a single feature
    df["capital"] = df["capital-gain"] - df["capital-loss"]
    df.drop(["capital-gain", "capital-loss"], axis=1, inplace=True)
    # df["capital"] = np.log1p(df["capital"].clip(lower=0))

    # freq = df["native-country"].value_counts()
    # df["native-country"] = df["native-country"].apply(
    #     lambda x: x if pd.notna(x) and freq.get(x, 0) >= 100 else "Other"
    # )
    df.drop("native-country", axis=1, inplace=True)

    # Convert target variable to binary
    df["income"] = df["income"].apply(lambda x: 1 if (x.strip() == ">50K") else 0)
    print(df.head())
    return df

df_train = data_preprocessing("adult/adult.data")
df_test = data_preprocessing("adult/adult.test")


# freq = df["occupation"].value_counts()
# df["occupation"] = df["occupation"].apply(
#     lambda x: x if pd.notna(x) and freq.get(x, 0) >= 100 else "Other"
# )

# sns.(x='income',y='sex',data=df)
# plt.show()

# sns.countplot(data=df, x="sex", hue="income")
# plt.title("Income count by sex")
# plt.show()

# sns.boxplot(x='income',y=np.log1p(df["capital"]),data=df)
# plt.show()
# exit()

# Separate features and target variable
X_train = df_train.drop("income", axis=1)
y_train = df_train["income"]
X_test = df_test.drop("income", axis=1)
y_test = df_test["income"]




# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# Columns
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

#Preprocessing Pipelines

## Numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("kbins", KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="uniform")),
    ("scaler", StandardScaler()),
])

## Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", KNeighborsClassifier())
])


# Hyperparameter tuning
param_grid = {
    "classifier__n_neighbors": [15,21,25,31,35,41,45],
    # "classifier__weights": ["uniform", "distance"],
    # "classifier__p": [1, 2],   # 1 = Manhattan, 2 = Euclidean,
    # "preprocessing__num__kbins__n_bins": [3,5,10,20,30],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=6)
grid.fit(X_train, y_train)

print("Best CV score:", grid.best_score_)
print("Best parameters:", grid.best_params_)

# # Best model
# vscode
# y_pred = best_model.predict(X_test)
# print(classification_report(y_test, y_pred))


# # Discretize into bins
# kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# df[['age']] = kbins.fit_transform(df[['age']])
# df[['hours-per-week']] = kbins.fit_transform(df[['hours-per-week']])
# df[['capital']] = kbins.fit_transform(df[['capital']])



# # Perform one-hot encoding
# encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# x_encoded = encoder.fit_transform(x[categorical_cols])

# # Convert the encoded features to a DataFrame
# encoded_df = pd.DataFrame(x_encoded, columns=encoder.get_feature_names_out(categorical_cols
# ))
# # Drop original categorical columns and concatenate the encoded features
# x = x.drop(categorical_cols, axis=1)
# x = pd.concat([x.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)


# ## Main Model
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)



# param_grid = {"n_neighbors": list(range(1, 31))}

# grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
# grid.fit(X_train, y_train)

# best_k = grid.best_params_["n_neighbors"]

# knn = KNeighborsClassifier(n_neighbors=best_k)
# knn.fit(X_train, y_train)

# from sklearn.metrics import classification_report

# y_pred = knn.predict(X_test)
# print(classification_report(y_test, y_pred))

