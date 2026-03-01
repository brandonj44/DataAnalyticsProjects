# import data file and transform into df through pandas
import pandas as pd
import numpy as np
import statsmodels.api as sm
df = pd.read_csv("C:/Users/velez/Downloads/medical_clean.csv")
df = pd.DataFrame(df)
df.info()

#categories look appropriate as is
print(df.Gender.unique())
print(df.Area.unique())
print(df.Marital.unique())
print(df.ReAdmis.unique())
print(df.Initial_admin.unique())
print(df.Complication_risk.unique())
print(df.Services.unique())
print(df.Stroke.unique())
print(df.Asthma.unique())
print(df.Soft_drink.unique())
print(df.Diabetes.unique())
print(df.Overweight.unique())
print(df.HighBlood.unique())
print(df.vitD_supp.unique())

#assess duplicates, all fields are false, nothing to change
print(df.duplicated().value_counts()) 

#ensure no NA values across df
df.dropna()

#Outlier Analysis
import seaborn

#%% initial days outliers

seaborn.boxplot(x='Initial_days',data=df)
# no outliers, does not need cleaned, just clean NA

df['Initial_days'].fillna(df['Initial_days'].median(),inplace=True)

#%% Age outliers assessment- in integer, no need to convert
seaborn.boxplot(x='Age',data=df)

# the box plot shows there are no outliers, so we will clean by replacing NA with median and move on
df['Age'].fillna(df['Age'].median(),inplace=True)

#summary statistics for dependent and ind variables
print(df.ReAdmis.describe(include='all'))

print(df.Age.describe(include='all'))
print(df.Gender.describe(include='all'))
print(df.Initial_admin.describe(include='all'))

print(df.Anxiety.describe(include='all'))
print(df.Initial_days.describe(include='all'))
print(df.Reflux_esophagitis.describe(include='all'))

print(df.Services.describe(include='all'))
print(df.Hyperlipidemia.describe(include='all'))
print(df.Diabetes.describe(include='all'))

print(df.Overweight.describe(include='all'))
print(df.Stroke.describe(include='all'))
print(df.HighBlood.describe(include='all'))

# univariate graphs
import seaborn as sns
import matplotlib.pyplot as plt

df['Anxiety'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Initial_admin'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Reflux_esophagitis'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Gender'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['ReAdmis'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Services'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Hyperlipidemia'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Diabetes'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Overweight'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Stroke'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['HighBlood'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Initial_days'].hist()
plt.xlabel('Initial_days')
plt.ylabel('Frequency')
plt.show()

df['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# BiVariate graphs

pd.crosstab(df["Initial_admin"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Gender"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Reflux_esophagitis"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Services"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Anxiety"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Hyperlipidemia"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Diabetes"], df["ReAdmis"]).plot.bar(rot=0)
plt.show()

pd.crosstab(df["Overweight"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["Stroke"], df["ReAdmis"]).plot.bar(rot=0)
plt.show() 

pd.crosstab(df["HighBlood"], df["ReAdmis"]).plot.bar(rot=0)
plt.show()

sns.barplot(x="ReAdmis", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="ReAdmis", y="Age", data=df)
plt.show() 

#Initial log regression model
X = df[['Reflux_esophagitis', 'Gender', 'Anxiety', 'Initial_admin','Age','Services', 'Stroke', 'Hyperlipidemia', 'Diabetes', 'Overweight', 'HighBlood', 'Initial_days']]
y = df['ReAdmis']

#build df with dummies
X = pd.get_dummies(data=X,drop_first=True)

y = pd.get_dummies(data=y,drop_first=True)

print(X.head(), y.head())

#one hot encode
y.replace(['Yes','No'],[1,0],inplace=True)
X.replace(['True','False'],[1,0],inplace=True)

# Xport
X.to_csv("C:/Users/velez/Downloads/InitialLogdummyfeatures.csv")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

#Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
model = LogisticRegression()
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)

print('Initial Y Intercept           :', model.intercept_)

print(model.predict_proba(X_train))
print(classification_report(y_train, model.predict(X_train)))

cm = confusion_matrix(y_train, model.predict(X_train))

print(cm)

#summary stats
model = sm.Logit(y_train,X_train)
result = model.fit()

result.summary()

#Feature importance visualization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

coefficients = model.coef_

avg_importance = np.mean(np.abs(coefficients), axis=0)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))


#Build Reduced model using only features with significance score over 0.75

X = X[['Initial_days', 'Initial_admin_Emergency Admission', 'Services_MRI', 'Stroke_Yes', 'Services_CT Scan', 'Gender_Nonbinary', 'HighBlood_Yes', 'Anxiety_Yes']]

#build df with dummies
X = pd.get_dummies(data=X,drop_first=True)

print(X.head())

#one hot encode
X.replace(['Yes','No'],[1,0],inplace=True)

# Xport dataset
X.to_csv("C:/Users/velez/Downloads/ReducedLogDummyfeatures.csv")

#train on data (use .ravel on y to transform into a 1D array)
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

#Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
model = LogisticRegression()
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print('Reduced Y Intercept           :', model.intercept_)

print(model.predict_proba(X_train))
print(classification_report(y_train, model.predict(X_train)))

cm = confusion_matrix(y_train, model.predict(X_train))

print(cm)

#CM visual
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Initial model: Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

X_train=np.asarray(X_train)

#reduced log model stats
model = sm.Logit(y_train,X_train)
result = model.fit()

result.summary()





