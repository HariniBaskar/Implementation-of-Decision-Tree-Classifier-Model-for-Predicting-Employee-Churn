# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Import the required packages. 

2)Read the data set. 

3)Apply label encoder to the non-numerical column inoreder to convert into numerical values. 

4)Determine training and test data set. 

5)Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harini.B
RegisterNumber: 212221230035
*/
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![pic1](https://github.com/HariniBaskar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427253/696b9b18-3f3e-4029-ad12-e80aafc2f35a)

![pic2](https://github.com/HariniBaskar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427253/dcb30ea3-6aa8-4cd1-9d46-84d245fe7d8f)

![pic3](https://github.com/HariniBaskar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427253/51d013f2-16fd-4a49-b023-0e9ee61f0256)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
