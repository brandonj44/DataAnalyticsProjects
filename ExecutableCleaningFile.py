# import data file and transform into df through pandas
import pandas as pd
df0 = pd.read_csv('C:/Users/velez/.spyder-py3/medical_raw_data.csv')
df = pd.DataFrame(df0)
df.info()

#determine percentage of missingness, then visualize
df_nullity = df.isnull()
print(df_nullity.mean() * 100 )
 
#install missingno required (already done on machine) and visually assess duplicates
import missingno as msno
msno.bar(df)

#%% assess duplicates
print(df.duplicated().value_counts()) 

# analyze numeric column outliers
#income outlier analysis
#%%install seaborn (on machine already)
import pandas as pd
import seaborn
seaborn.boxplot(data = df, x='Income')

#convert str to float 

df['Income'] = df['Income'].astype(float)

#%%
#convert all income outliers to median value : outlier count 10000-7206 = 2794 , visual cutoff at 100k

import numpy as np
df['Income'] = np.where(df['Income']>100000,np.nan,df['Income'])
df.Income.info()

df['Income'].fillna(df['Income'].median(), inplace= True)
df.Income.info()

#%%TotalCharge
import pandas as pd
import seaborn
seaborn.boxplot(data = df, x='TotalCharge')

#%%convert str to float

df['TotalCharge'] = df['TotalCharge'].astype(float)

#%% outliers visually begin at 14k, cut off there
#convert all TotalCharge outliers to median value, outlier count: 10000-9528 = 472

import numpy as np
df['TotalCharge'] = np.where(df['TotalCharge']>14000,np.nan,df['TotalCharge'])
df.TotalCharge.info()

df['TotalCharge'].fillna(df['TotalCharge'].median(), inplace= True)
df.TotalCharge.info()

#%%AdditionalCharge
import pandas as pd
import seaborn
seaborn.boxplot(data = df, x='Additional_charges')

#%%ensure we have float 
df['Additional_charges'] = df['Additional_charges'].astype(float)

# outliers are above 27k

import numpy as np
df['Additional_charges'] = np.where(df['Additional_charges']>27000,np.nan,df['Additional_charges'])
df.Additional_charges.info()

#convert all additional chargers outliers to median value

df['Additional_charges'].fillna(df['Additional_charges'].median(), inplace= True)
df.Additional_charges.info()

# 10k-9568 =432 outliers changed

#%% initial days 
import pandas as pd
import seaborn
seaborn.boxplot(x='Initial_days',data=df)
# no outliers, does not need cleaned, just clean NA
import numpy as np
df['Initial_days'].fillna(df['Initial_days'].median(),inplace=True)

#%% VitD, already in float
import pandas as pd
import seaborn
seaborn.boxplot(x='VitD_levels',data=df)

#%% The boxplot shows a cutoff at a high of 24 and a low of 11, so values outside that range must be removed
# there are shown to be 528 outliers above the value 24
import numpy as np
df['VitD_levels'] = np.where(df['VitD_levels']>24,np.nan,df['VitD_levels'])
df.VitD_levels.info()

df['VitD_levels'].fillna(df['VitD_levels'].median(), inplace= True)
df.VitD_levels.info()

#%% outliers BELOW: There are 10 replaced values below the value of 11
import numpy as np
df['VitD_levels'] = np.where(df['VitD_levels']<11,np.nan,df['VitD_levels'])
df.VitD_levels.info()

df['VitD_levels'].fillna(df['VitD_levels'].median(), inplace= True)
df.VitD_levels.info()

#%% Age outliers assessment- in integer, no need to convert
import pandas as pd
import seaborn
seaborn.boxplot(x='Age',data=df)

#%% the box plot shows there are no outliers, so we will clean by replacing NA with median and move on
import numpy as np
df['Age'].fillna(df['Age'].median(),inplace=True)

#%% amount of Children outliers assessment, no string conversion necessary
import pandas as pd
import seaborn
seaborn.boxplot(x='Children',data=df)

#%% Box plot shows outliers begin when values increase above 7, however only 3 values are present. 
# I will not adjust outliers, but will ensure NA is imputed with median
import numpy as np
df['Children'].fillna(df['Children'].median(),inplace=True)


#%% Population outlier assessment
import pandas as pd
import seaborn
seaborn.boxplot(x='Population',data=df)

#%% outliers are above the value 26,000, however these should not be erased as population outlier data could be pertinent to medical decisions
# remove NA with median, as this information exists for all patients but is likely incomplete on surveys
import numpy as np
df['Population'].fillna(df['Population'].median(),inplace=True)

#%% Doc_visits outlier assessment
import pandas as pd
import seaborn
seaborn.boxplot(x='Doc_visits',data=df)
#%% range from 1-9 with no outliers, so let's impute NA with avgs
import numpy as np
df['Doc_visits'].fillna(df['Doc_visits'].median(),inplace=True)

#%% Full_meals_eaten outlier assessment
import pandas as pd
import seaborn
seaborn.boxplot(x='Full_meals_eaten',data=df)

#%% plot ranges from 0-5 and has 2 outliers above the value of 5. We can leave these and deal with NA
import numpy as np
df['Full_meals_eaten'].fillna(df['Full_meals_eaten'].median(),inplace=True)


#%%convert indicator columns 1/0 to yes/no and default NaN to No- assume patients do not have conditions unless told so
df['Anxiety'].replace(to_replace = [1,0,'nan'],value = ['Yes','No','No'], inplace=True)
df['Anxiety'].fillna("No", inplace=True)
df['Anxiety'].unique()
#%%
#same for Overweight
df['Overweight'].replace(to_replace = [1,0,'nan'],value = ['Yes','No','No'], inplace=True)
df['Overweight'].fillna("No", inplace=True)
df['Overweight'].unique()
#%%
#Soft drink has NAs to replace
df['Soft_drink'].replace(to_replace = ['nan'],value = ['No'], inplace=True)
df['Soft_drink'].fillna("No", inplace=True)
df['Soft_drink'].unique()


#%% catchall for any numeric fields 
df.fillna(df.median(numeric_only=True), inplace=True)

#%%For categorical re-expression, we will analyze the unique values present in categorical columns and look for redundancies/typos.
print(df.Gender.unique())
print(df.Area.unique())
print(df.Timezone.unique())
print(df.Education.unique())
print(df.Marital.unique())
print(df.Employment.unique())
print(df.ReAdmis.unique())
print(df.Soft_drink.unique())
print(df.Initial_admin.unique())
print(df.Complication_risk.unique())
print(df.Services.unique())

df.to_csv(r"C:\Users\velez\Downloads\Cleaned_Data_D206.csv")

#%%import packages for PCA
import pandas as pd
df = pd.read_csv("C:\\Users\\velez\\Downloads\\Cleaned_Data_D206.csv")
print(df.head())
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Define PCA Variables

data = df[['Income','VitD_levels','Initial_days','TotalCharge','Additional_charges']]
data_normalized=(data-data.mean())/data.std()

# Normalize data
pca = PCA(n_components=data.shape[1])

pca.fit(data_normalized)
data_pca = pd.DataFrame(data_normalized)
columns = ['PC1', 'PC2','PC3','PC4','PC5']

loadings = pd.DataFrame(pca.components_.T,
        columns= ['PC1', 'PC2','PC3','PC4','PC5'],
        index = data.columns)
#%% my environment doesn't show full PCA results so we export to view
loadings.to_csv(r"C:\Users\velez\Downloads\PCA.csv")

# scree plot for eigenvalues
#create covariance matrix
cov_matrix = np.dot(data_normalized.T, data_normalized) / data.shape[0]
# define eigenvalues
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]

#now create scree plot
plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalue')
plt.axhline(y=1, color = 'red')
plt.show()
