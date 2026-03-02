
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/velez/OneDrive/Desktop/School/D212/DataSets/medical_clean.csv")
print(data.head())

cont_df=data[['Initial_days','Income','VitD_levels']]

#scale data
scaler=StandardScaler()
norm_df=scaler.fit_transform(cont_df)

scaled_df=pd.DataFrame(norm_df,columns=cont_df.columns)

scaled_df.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 2/Scaled_df.csv")
scaled_df.head()

from sklearn.decomposition import PCA
#initiate and define PCA
pca=PCA(n_components=3)
PC = pca.fit_transform(scaled_df)
#define and export loadings matrix
loading_matrix=pd.DataFrame(pca.components_,columns=cont_df.columns,index=('PC0','PC1', 'PC2'))
print(loading_matrix)

exp_var = pca.explained_variance_ratio_
print("explained variance ratio: ", exp_var)

#np array beginning with index 1
pcomp = np.arange(pca.n_components_) + 1

#Scree plot elbow method
plt.figure(figsize=(13,6))
plt.plot(pcomp,
         exp_var,
         'b-')
plt.title('Scree plot (Elbow method)', fontsize=16)
plt.xlabel('Number of components',fontsize=16)
plt.ylabel('Variance proportion',fontsize=16)
plt.grid()
plt.show()

#top 2 PC var
print(dict(zip(['PC1','PC2'], pcomp)))
print("Var of top 2 PCs:", pca.explained_variance_[:2])

total_var = np.sum(pcomp[:2])/np.sum(pcomp)
print("total variance: ",total_var)

var = pca.explained_variance_
print("explained variance: ", var)

#plot kaiser criterion
plt.figure(figsize=(13,6))
plt.plot(pcomp,
        var,
        'b')
plt.title('Scree Plot Kaiser criterion', fontsize=16)
plt.xlabel('Number of components', fontsize=16)
plt.ylabel('eigenvalues',fontsize=16)
plt.axhline(y=1, color='g', linestyle='dashdot')
plt.grid()
plt.show()

# 2 components
print(dict(zip(['PC0','PC1'],pcomp)))

#variance of each 3
print("Variance of first 2 PCs:", pca.explained_variance_[:2])
#variance of total 2
total_var_capt = np.sum(pca.explained_variance_ratio_[:2])
print("total variance: ", total_var_capt)
#plot var ratio
component = range(pca.n_components_)
plt.bar(component, pca.explained_variance_ratio_, color='red')
plt.xlabel('Principal Component')
plt.ylabel('Variance %')














