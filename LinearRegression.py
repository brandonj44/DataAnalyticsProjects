# import data file and transform into df through pandas
import pandas as pd
import numpy as np
df0 = pd.read_csv("C:/Users/velez/Downloads/medical_clean.csv")
df = pd.DataFrame(df0)
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

#assess duplicates, all fields are false
print(df.duplicated().value_counts()) 

#ensure no NA values across df
df.dropna()

#Outlier Analysis

#%%TotalCharge
import seaborn
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


#%%summary statistics for dependent and ind variables
print(df.Initial_admin.value_counts())
print(df.Complication_risk.value_counts())
print(df.Age.describe(include='all'))

print(df.Services.value_counts())
print(df.Stroke.value_counts())
print(df.Asthma.value_counts())

print(df.Soft_drink.value_counts())
print(df.Diabetes.value_counts())
print(df.Overweight.value_counts())

print(df.HighBlood.value_counts())
print(df.vitD_supp.value_counts())
print(df.Initial_days.describe(include='all'))


# univariate graphs
import seaborn as sns
import matplotlib.pyplot as plt

df['Initial_admin'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Complication_risk'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

df['Services'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Asthma'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Stroke'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Soft_drink'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Diabetes'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Overweight'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['vitD_supp'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['HighBlood'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['ReAdmis'].value_counts().sort_index().plot.bar()
plt.ylabel('Frequency')
plt.show()

df['Initial_days'].hist()
plt.xlabel('Initial_days')
plt.ylabel('Frequency')
plt.show()

df['TotalCharge'].plot.hist()
plt.xlabel('TotalCharge')
plt.ylabel('Frequency')
plt.show()

# BiVariate graphs

sns.barplot(x="Initial_admin", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Complication_risk", y="Initial_days", data=df)
plt.show() 

plt.scatter(x="Age", y="Initial_days", data=df)
plt.xlabel('Age')
plt.ylabel('Initial_days')
plt.show() 

sns.barplot(x="Services", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Stroke", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Asthma", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Soft_drink", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Diabetes", y="Initial_days", data=df)
plt.show()

plt.scatter(x="TotalCharge", y="Initial_days", data=df)
plt.xlabel('TotalCharge')
plt.ylabel('Initial_days')
plt.show() 

sns.barplot(x="HighBlood", y="Initial_days", data=df)
plt.show()

sns.barplot(x="vitD_supp", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="Overweight", y="Initial_days", data=df)
plt.show() 

sns.barplot(x="ReAdmis", y="Initial_days", data=df)
plt.show() 

#Initial multiple linear regression model
X0 = df[['Initial_admin', 'Income', 'TotalCharge', 'ReAdmis', 'Complication_risk', 'Age','Services', 'Stroke','Asthma', 'Diabetes', 'Overweight', 'HighBlood', 'vitD_supp']]
Y = df['Initial_days']
#build df with dummies
X = pd.get_dummies(data=X0,drop_first=True)
print(df.head())

#one hot encode
X.replace(['Yes','No'],[1,0],inplace=True)
X.replace([True , False],[1,0],inplace=True)

#Export cleaned data set 
#X.to_csv("C:/Users/velez/Downloads/InitialModelFeaturesProj1BJV.csv")

#train on data
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#build LR
lr = LinearRegression()
lr.fit(X, Y)

#compare Y values predicted to actual
y_pred = lr.predict(X)

residuals= Y - y_pred
print("Y variable Residuals    :", residuals)
plt.scatter(residuals,y_pred)
plt.xlabel('Initial Model Residual Values')
plt.ylabel('Y Prediction Value')
plt.show() 

df2=pd.DataFrame(data={'Predictions':y_pred,'Actuals':Y})
print(df2.head())

#model evaluation and metrics
print('Initial model R Square        :', lr.score(X,Y))
print('Initial Y Intercept           :', lr.intercept_)
print('Initial Mean Absolute Error   :', metrics.mean_absolute_error(Y , y_pred))
print('Initial Mean Squared Error    :', metrics.mean_squared_error(Y , y_pred))
print('Residual Standard Error       :',np.sqrt(metrics.mean_squared_error(Y , y_pred)))
print('Initial Coefficients          :', lr.coef_)

result = sm.OLS(Y, X).fit()

print(result.summary())

#look at correlation matrix to see iffeatures are redundant/ homoscedatic
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(X.corr(),annot=True,ax=ax)
plt.show()

print("Correlations   :", X.corr())

#Forward feature selection
import statsmodels.api as sm

def forward_selection(X, y):
    selected_features = []
    while True:
        remaining_features = [f for f in X.columns if f not in selected_features]
        new_pval = pd.Series(index=remaining_features)
        for feature in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            new_pval[feature] = model.pvalues[feature]
        min_pval = new_pval.min()
        if min_pval < 0.05: 
            selected_features.append(new_pval.idxmin())
        else:
            break
    return selected_features

selected = forward_selection(X, Y)
print(f"Selected features: {selected}")

#Build reduced lr model from columns containing the 7 strongest features
X1 = X[['TotalCharge', 'Initial_admin_Emergency Admission', 'Complication_risk_Medium', 'Complication_risk_Low', 'HighBlood_Yes', 'Diabetes_Yes', 'ReAdmis_Yes']]
Y = df['Initial_days']
print(df.head())


lr = LinearRegression()
lr.fit(X1, Y)

#build df to compare Y values predicted to actual
y_pred = lr.predict(X1)

residuals= Y - y_pred
print("Y variable Residuals    :", residuals)
plt.scatter(residuals, y_pred)
plt.xlabel('Adjusted Model Residual Value')
plt.ylabel('Y Prediction Value')
plt.show() 
 
df2=pd.DataFrame(data={'Predictions':y_pred,'Actuals':Y})
print(df2.head())

print('Reduced model R Square            :', lr.score(X1,Y))
print('Reduced model Intercept           :', lr.intercept_)
print('Reduced model Mean Absolute Error :', metrics.mean_absolute_error(Y , y_pred))
print('Reduced model Mean Squared Error  :', metrics.mean_squared_error(Y , y_pred))
print('Reduced Residual Standard Error   :',np.sqrt(metrics.mean_squared_error(Y , y_pred)))
print('Reduced model coefficients        :', lr.coef_)


result = sm.OLS(Y, X).fit()

print(result.summary())
