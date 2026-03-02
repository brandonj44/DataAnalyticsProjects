
# libraries and packages
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# get csv and turn into pandas df
df = pd.read_csv("C:/Users/velez/OneDrive/Desktop/School/D212/DataSets/medical_clean.csv")
print(df.head())

# Define X as selected variables for cluster analysis
X = df[['Initial_days','Income','VitD_levels']].round(2)

# Numeric variables must be scaled for kmeans clustering using standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Save clean dataset, back to pd df to export
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 1/X_scaled.csv")

# prepare elbow method to select K number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=None)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#define plot visuals
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow chart for K value selection')
plt.xlabel('K value')
plt.ylabel('WCSS')
plt.show()

# fit K Means using recommended K value
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# define kmeans labels as column into original dataset for use in clustering
df['Cluster'] = kmeans.labels_
df.head

# Build cluster using scatterplot by population and initial days
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Income'], y=df['Initial_days'], hue=df['Cluster'], palette='colorblind')
plt.scatter(kmeans.cluster_centers_, kmeans.cluster_centers_, s=300, c='red', label='Centroids')
plt.title('Initial Days by Income')
plt.legend()
plt.show()

#KMeans
from sklearn.cluster import KMeans
k_model = KMeans(n_clusters=3, n_init=25, random_state=300)

#fit Kmeans on scaled data
k_model.fit(X_scaled)

#model evaluation using model labels and silh
evaluate = pd.Series(k_model.labels_).value_counts()
print(evaluate)

from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(X_scaled, k_model.labels_)
print("Model Silhouette Score: ",silhouette_score)


#Analyze/Interpret results
Fin_model=KMeans(n_clusters=3, n_init=25, random_state=300)
Fin_model.fit(X_scaled)

centroid=pd.DataFrame(Fin_model.cluster_centers_,
                       columns=['Income', 'Initial_days','VitD_levels'])
print(centroid)

plt.figure(figsize=(12,10))

ax=sns.scatterplot(data=df,
                   x = 'Income',
                   y = 'Initial_days',
                   hue = Fin_model.labels_,
                   palette = 'colorblind',
                   alpha= 0.9,
                   s=200,
                   legend=True)

ax=sns.scatterplot(data=centroid,
                   x = 'Income',
                   y = 'Initial_days',
                   palette = 'colorblind',
                   alpha= 0.9,
                   s=200,
                   legend=True)




