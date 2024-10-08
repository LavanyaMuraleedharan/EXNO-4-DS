# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
 ```
![image](https://github.com/user-attachments/assets/03f5b834-e866-4de4-8452-ac113cf1b8b2)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/13f5e8d3-6bde-468f-91af-cdd282d14d98)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/d5584636-9db1-4998-92cf-1e24f74650ef)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/bbe0928e-eb42-4748-9a5a-984bff67e0de)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/eb6726db-3497-4f21-951d-2e05425a3128)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/bb9715f3-f788-4f25-ab6b-88ea837c936f)
```
data2
```
![image](https://github.com/user-attachments/assets/1f818126-5d1b-4206-a147-bc1ffe89a72a)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/2198a85c-e532-4f02-bf20-68eecc8716f2)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/f9fac427-fe90-48f2-a19c-339137f56f1b)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/ee8155c5-af92-4858-863a-a8ce9669e1f2)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/a5d4b9b5-f8e8-4b06-a019-45341ce20982)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/0b4942c7-7abb-45f2-93d8-c462dd329faf)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/5f147e22-3ddc-4005-bba1-93c3753f73af)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/2e42f0d2-7215-422e-abcc-c084e0f2f9fd)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/1df6dcd3-b0f3-4284-b505-f4eccf7dbab0)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/d55eb5f4-522c-4ce9-b471-79559fcd60b8)
```
data.shape
```
![image](https://github.com/user-attachments/assets/5c88e641-3a39-417a-b2bb-477108334be1)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/f06f4760-4b44-4729-8b71-56922f7b89ff)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/a669f267-d9d0-4795-b172-be397325c031)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/0d8e7193-7dfe-4649-8933-c7d4d67d2b06)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/727dc94b-d71d-456d-a74a-9d8d3c2e2cfc)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/fe6564c3-8114-4fbe-9c85-6be4da70e6b0)



# RESULT:
 Thus, Feature selection and Feature scaling has been used on the given dataset.


