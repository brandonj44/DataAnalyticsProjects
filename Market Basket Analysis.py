#packages
import pandas as pd
from pandas import DataFrame
import numpy as np
#import and look at shape
data=pd.read_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/medical_market_basket.csv")
print(data.head())
print(data.shape)

#remove entirely NA rows
data=data[data['Presc01'].notna()]
data.shape

from mlxtend.preprocessing import TransactionEncoder
#Convert df to list of lists
rows=[]
for i in range (0,7501):
    rows.append([str(data.values[i,j])
for j in range(0,20)])

#call TransEnc and input above list
DE= TransactionEncoder()
array=DE.fit(rows).transform(rows)

#now we can use as df
transaction = pd.DataFrame(array, columns = DE.columns_)

print(transaction.head())

#display items as columns
for col in transaction.columns:
    print(col)

#drop NaN
clean_df = transaction.drop(['nan'], axis=1)
clean_df.head(7501)
#save clean dataset
clean_df.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/df_clean.csv",index=False)

#  MBA packages
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = clean_df

# top 5 used meds
count = data.loc[:,:].sum()
pop_item = count.sort_values( ascending=False).head(5)
pop_item = pop_item.to_frame()
pop_item = pop_item.reset_index()
pop_item = pop_item.rename(columns= {'index': 'Products', 0: 'count'})
print(pop_item)

#apriori alg
rules = apriori(data, min_support = 0.02, use_colnames=True)
print(rules.head(5))

rul_table = association_rules(rules, metric='lift', min_threshold=1)
print(rul_table.head(20))

rul_table.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/rul_table.csv",index=False)

top_three_rules_conf = rul_table.sort_values('confidence', ascending=False).head(3)
print(top_three_rules_conf)

top_three_rules_conf.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/top_three_rules_conf.csv",index=False)

top_three_rules_lift = rul_table.sort_values('lift', ascending=False).head(3)
print(top_three_rules_lift)

top_three_rules_lift.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/top_three_rules_lift.csv",index=False)

top_three_rules_supp = rul_table.sort_values('support', ascending=False).head(3)
print(top_three_rules_supp)

top_three_rules_supp.to_csv("C:/Users/velez/OneDrive/Desktop/School/D212/Task 3/top_three_rules_Supp.csv",index=False)



