import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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

#%% Population outlier assessment

seaborn.boxplot(x='Population',data=df)

# outliers are above the value 26,000, however these should not be erased as population outlier data could be pertinent to medical decisions
# remove NA with median, as this information exists for all patients but is likely incomplete on surveys
df['Population'].fillna(df['Population'].median(),inplace=True)

#%% Select variables for KNN model
X = df[['Age', 'Population', 'Marital', 'Gender', 'VitD_levels', 'Soft_drink', 'HighBlood', 'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'Anxiety', 'Asthma', 'Services']]
y = df["Stroke"]

X = pd.get_dummies(data=X,drop_first=False)
y = pd.get_dummies(data=y,drop_first=False)

#one hot encode
X.replace([True,False],[1,0],inplace=True)
y.replace([True,False],[1,0],inplace=True)

#drop redundant binary columns
X = X.drop(['Soft_drink_No','HighBlood_No','Overweight_No','Arthritis_No','Diabetes_No','Hyperlipidemia_No','Anxiety_No','Asthma_No'], axis=1)
y = y.drop(['No'], axis=1)
y = y.rename({'Yes' :'Stroke'},axis=1)

#Standardize scaling for KNN on numeric columns
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X['Age'] = scale.fit_transform(X[['Age']])
X['Population'] = scale.fit_transform(X[['Population']])
X['VitD_levels'] = scale.fit_transform(X[['VitD_levels']])

#redefine as df
X = pd.DataFrame(X)

X.to_csv("C:/Users/velez/Downloads/D209CleanedDataSet.csv")

#create, train, and export training data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,
test_size=0.2, random_state=15, stratify=y)

X_train = pd.DataFrame(X_train, columns = X_train.columns)
X_train.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/X_train.csv")

X_test = pd.DataFrame(X_test, columns = X_test.columns)
X_test.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/X_test.csv")

y_train = pd.DataFrame(y_train, columns = y_test.columns)
y_train.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/y_train.csv")

y_test = pd.DataFrame(y_test, columns = y_test.columns)
y_test.to_csv("C:/Users/velez/OneDrive/Desktop/School/D209/y_test.csv")

#KNN Analysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


#Build pipeline, define steps (removed scaling as this was done preprocessing)
steps = [('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors' : np.arange(1,25)}

#Select parameters for KNN
knncv = GridSearchCV(estimator = pipeline,
                     param_grid = parameters,
                     n_jobs = -1,
                     cv = 5)

#fit newly defined model with training data
y_train=np.asarray(y_train)
knncv.fit(X_train, y_train.ravel())

#Accuracy calculations
print("Training Accuracy: ", knncv.score(X_train, y_train.ravel()))
print("Testing Accuracy: ", knncv.score(X_test, y_test))

#KNN model metrics
knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train.ravel())
print("KNN Model Accuracy: ")
print(knn.score(X_test, y_test))
y_predicted = knn.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_predicted))
y_predicted_probability = knn.predict_proba(X_test)[:,1]
print("KNN model AUC: ")
print(roc_auc_score(y_test, y_predicted_probability))
