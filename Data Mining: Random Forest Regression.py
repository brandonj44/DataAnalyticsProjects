import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as r2

df = pd.read_csv("C:/Users/velez/OneDrive/Desktop/School/D209/medical_clean.csv")
df = pd.DataFrame(df)
df.info()

#assess duplicates, all fields are false
print(df.duplicated().value_counts()) 

#categories look appropriate as is
print(df.Gender.unique())
print(df.Area.unique())
print(df.Marital.unique())

print(df.ReAdmis.unique())
print(df.Initial_admin.unique())
print(df.Complication_risk.unique())

print(df.Services.unique())
print(df.Arthritis.unique())
print(df.Hyperlipidemia.unique())

print(df.Asthma.unique())
print(df.Anxiety.unique())
print(df.Stroke.unique())

print(df.Soft_drink.unique())
print(df.Diabetes.unique())
print(df.Overweight.unique())
print(df.HighBlood.unique())

#ensure no NA values across df
df.dropna()

#Outlier Analysis
import seaborn

#%% VitD, already in float
seaborn.boxplot(x='VitD_levels',data=df)

# The boxplot shows a cutoff at a high of 24 and a low of 11, so values outside that range must be removed
# there are shown to be 528 outliers above the value 24

df['VitD_levels'] = np.where(df['VitD_levels']>24,np.nan,df['VitD_levels'])
df.VitD_levels.info()

df['VitD_levels'].fillna(df['VitD_levels'].median(), inplace= True)
df.VitD_levels.info()

# outliers BELOW: There are 10 replaced values below the value of 11
df['VitD_levels'] = np.where(df['VitD_levels']<11,np.nan,df['VitD_levels'])
df.VitD_levels.info()

df['VitD_levels'].fillna(df['VitD_levels'].median(), inplace= True)
df.VitD_levels.info()

#%% Age outliers assessment- in integer, no need to convert
seaborn.boxplot(x='Age',data=df)

# the box plot shows there are no outliers, so we will clean by replacing NA with median and move on
df['Age'].fillna(df['Age'].median(),inplace=True)

#%%TotalCharge
seaborn.boxplot(data = df, x='TotalCharge')

#convert str to float
df['TotalCharge'] = df['TotalCharge'].astype(float)

# outliers visually begin at 14k, cut off there

#convert all TotalCharge outliers to median value, outlier count: 10000-9528 = 472
df['TotalCharge'] = np.where(df['TotalCharge']>14000,np.nan,df['TotalCharge'])
df.TotalCharge.info()

df['TotalCharge'].fillna(df['TotalCharge'].median(), inplace= True)
df.TotalCharge.info()

#%% initial days outliers
seaborn.boxplot(x='Initial_days',data=df)
# no outliers, does not need cleaned, just clean NA
df['Initial_days'].fillna(df['Initial_days'].median(),inplace=True)

#%% Select variables
X = df[['Initial_admin', 'Income', 'TotalCharge', 'ReAdmis', 'Complication_risk', 'Age','Services', 'Stroke','Asthma', 'Diabetes', 'Overweight', 'HighBlood', 'VitD_levels']]
y = df['Initial_days']
#dummies for categories
X = pd.get_dummies(data=X,drop_first=False)

#one hot encode
X.replace([True,False],[1,0],inplace=True)

#drop redundant binary columns
X = X.drop(['ReAdmis_No','Stroke_No','Asthma_No','Diabetes_No', 'Overweight_No', 'HighBlood_No' ], axis=1)

#heatmap for multicollinearity
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,20))
seaborn.heatmap(X.corr(),annot=True,ax=ax)

X.to_csv("C:/Users/velez/Downloads/D209Proj2CleanedDataSet.csv")

y.to_csv("C:/Users/velez/Downloads/y.csv")

#create, train, and export training data
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,
test_size=0.2, random_state=15)

X_train = pd.DataFrame(X_train, columns = X_train.columns)
X_train.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/Task 2/X_train.csv")

X_test = pd.DataFrame(X_test, columns = X_test.columns)
X_test.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/Task 2/X_test.csv")

y_train = pd.DataFrame(y_train, columns = y_train.columns)
y_train.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/Task 2/y_train.csv")

y_test = pd.DataFrame(y_test, columns = y_test.columns)
y_test.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/Task 2/y_test.csv")

#define parameters to be used in gridsearch
parameters = {"n_estimators": [10,50,100],
              "max_features": [2,3,4],
              "max_depth": [8, None]
              }
#call random forest
rfr = RandomForestRegressor(n_estimators=100, max_depth=3,random_state=15)

#fit rf on train data
y_train = np.asarray(y_train)
rfr.fit(X_train,y_train.ravel())

#use now trained rf model and defined params to build gridsearch
cv = GridSearchCV(rfr, cv = 3, param_grid=parameters,verbose = 3)

#fit model using train data
cv.fit(X_train,y_train.ravel())

#use gridsearchcv to build y pred
y_pred = cv.predict(X_test)

#model metrics
print("MSE: ", MSE(y_test,y_pred))
print("Root MSE: ", MSE(y_test,y_pred)**1/2)
print("R Square: ", r2(y_test,y_pred))

#for potential model tuning
print("Best params suggestion: ", cv.best_params_)

print("cv.best score: ",cv.best_score_)

#populate list of feature names for use in tree
X_train.columns

#build decision tree visual
from sklearn import tree
plt.figure(figsize=(15,8))
tree.plot_tree(rfr.estimators_[0], feature_names=['Income', 'TotalCharge', 'Age', 'VitD_levels',
       'Initial_admin_Elective Admission', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission', 'ReAdmis_Yes',
       'Complication_risk_High', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Services_Blood Work', 'Services_CT Scan',
       'Services_Intravenous', 'Services_MRI', 'Stroke_Yes', 'Asthma_Yes',
       'Diabetes_Yes', 'Overweight_Yes', 'HighBlood_Yes'], class_names=True,precision=1,fontsize=7 )

