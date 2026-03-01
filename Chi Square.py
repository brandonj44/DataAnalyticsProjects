# Chi square
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

#pandas makes life ez
df = pd.read_csv(r"C:\Users\velez\OneDrive\Desktop\School\D207\medical_clean.csv", delimiter=',')

df = pd.DataFrame(df)

print(df.head())

#Create crosstab and contingency table
contingency = pd.crosstab(df['ReAdmis'], df['Stroke'])
print(contingency)

contingency_pct = pd.crosstab(df['ReAdmis'], df['Stroke'], normalize= 'index')
print(contingency_pct)

plt.figure(figsize=(20,18))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")

#chi square test. take 'contingency' as defined above and analyze
c, p, dof, expected = chi2_contingency(contingency)
print(p)

#interpreting p value
alpha = 0.05
print("p value is " + str(p))
if p <- alpha:
    print('Dependent(reject H0')
else:
    print('Independent (accept H0)')
    
#%% Part C, univariate graphs
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Users\velez\OneDrive\Desktop\School\D207\medical_clean.csv", delimiter=',')

df = pd.DataFrame(df)

df['Complication_risk'].value_counts().sort_index().plot.bar()
plt.show()

df['Services'].value_counts().sort_index().plot.bar()
plt.show()

sns.distplot(df.Income)
plt.show()
df.Income.describe()

sns.distplot(df.VitD_levels)
plt.show()
df.VitD_levels.describe()

# Part D, BiVariate graphs
df.plot(x = 'Initial_days', y = 'TotalCharge', kind = 'scatter');
plt.show() 
df.Initial_days.describe()
df.TotalCharge.describe()

pd.crosstab(df['HighBlood'], df['Overweight'], normalize= 'index').plot(kind="bar", stacked=True)
plt.show() 

