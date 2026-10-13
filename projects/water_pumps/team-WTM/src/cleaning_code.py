#loading dataset
import pandas as pd
train_label = pd.read_csv("../data/raw/Training_set_labels.csv")
train_values = pd.read_csv("../data/raw/Training_set_values.csv")

#merging dataset
train_data = train_values.merge(
    train_label,
    how="outer",
    on="id",
    sort=True,
    indicator=True
)

train_data["_merge"].value_counts()

train_data = train_data.drop(columns="_merge")

#2. Checking the distribution of the target variable:
train_data["status_group"].value_counts()

#missing values check
missing = train_data.isnull().sum().sort_values(ascending=False)
print(missing)

#checking duplicated rows ignoring the id column
train_data.drop(columns="id").duplicated().sum()

# Statistical summary
#overview of categorical columns.
categorical_columns = train_data.select_dtypes(include=["str"]).columns.tolist()
categorical_columns.remove("date_recorded")
print(categorical_columns)

#To visualise the uniques values in the categorical variables
for column in categorical_columns:
    print("\n", column)
    print(train_data[column].unique())

#overview of numerical columns
numeric_columns = train_data.select_dtypes(include=["int64", "float64"]).columns
print(numeric_columns)

train_data[numeric_columns].describe()

#replacing O values in population with unknown
import numpy as np
train_data["population"] = train_data["population"].replace(0, np.nan)
train_data["construction_year"] = train_data["construction_year"].replace(0, np.nan)


# convert date_recorded from string to datatime type
train_data["date_recorded"] = pd.to_datetime(train_data["date_recorded"])


#to drop any column, especially for the columns that have related values in the categorical variables
train_data = train_data.drop(columns=["num_private"])




