<H3>ENTER YOUR NAME: PANDEESWARAN N</H3>
<H3>ENTER YOUR REGISTER NO: 212224230191</H3>
<H3> DATE : 27.01.2026</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import libraries
```PYTHON
import pandas as pd
import numpy as np
import seaborn as sns   # for outlier detection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

### Read the dataset directly
```PYTHON
df = pd.read_csv('Churn_Modelling.csv')
print("First 5 rows of the dataset:")
df.head()
```

### Find missing values
```PYTHON
print(df.isnull().sum())
```

### Identify categorical columns
```PYTHON
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns:", categorical_cols.tolist())
```

### Apply Label Encoding to categorical columns
```PYTHON
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nData after encoding:")
print(df.head(5))
```
### Handling missing values only for numeric columns
```PYTHON
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean().round(1), inplace=True)

df.isnull().sum()
```

### Detect Outliers (example using seaborn)
```PYTHON
print("\nDetecting outliers (example: CreditScore column):")
sns.boxplot(x=df['CreditScore'])
```

### Example statistics for 'CreditScore'
```PYTHON
print("\nStatistics for 'CreditScore':")
df['CreditScore'].describe()
```

### Splitting features (X) and labels (y)
```PYTHON
X = df.drop('Exited', axis=1).values  # Features (drop target column)
y = df['Exited'].values   

print("\nFeature Matrix (X):")
print(X)
print("\nLabel Vector (y):")
print(y)
```
### Normalizing the features
```PYTHON
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### First 5 rows after normalization
```PYTHON
pd.DataFrame(X_normalized, columns=df.columns[:-1]).head()
```

### Splitting into Training and Testing Sets
```PYTHON
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

print("\nShapes of Training and Testing sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

## OUTPUT:

<img width="1169" height="252" alt="image" src="https://github.com/user-attachments/assets/6334b160-ed05-4b8d-8956-739fc8e0ce5a" />
<img width="1151" height="377" alt="image" src="https://github.com/user-attachments/assets/9a2340e7-51cf-4b6e-8185-cbc08bd47d03" />
<img width="1190" height="540" alt="image" src="https://github.com/user-attachments/assets/baefdcf5-1807-40fd-83b4-5af4e08cc871" />
<img width="1202" height="381" alt="image" src="https://github.com/user-attachments/assets/81230442-1fb9-46b1-ab3c-dd908d08db19" />
<img width="1200" height="765" alt="image" src="https://github.com/user-attachments/assets/33c2825d-1b54-41c6-b35b-f398328417b4" />
<img width="1202" height="453" alt="image" src="https://github.com/user-attachments/assets/6bb00579-e3fd-455e-ac90-3bfaa38e6a55" />
<img width="1201" height="740" alt="image" src="https://github.com/user-attachments/assets/1e516f9b-5025-492d-bcaf-be23731d22f6" />
<img width="1181" height="341" alt="image" src="https://github.com/user-attachments/assets/b3358be5-0a2a-4bc9-b1c0-7076f1aa9f83" />
<img width="1188" height="576" alt="image" src="https://github.com/user-attachments/assets/e0652aac-767f-4909-957c-5569dd398db1" />
<img width="1198" height="731" alt="image" src="https://github.com/user-attachments/assets/050bc24d-a3ec-4937-a073-78c5fcc37fa1" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
