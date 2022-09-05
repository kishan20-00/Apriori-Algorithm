import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df= pd.read_excel('online_retail_2.xlsx')
df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['Invoice'], inplace=True)
df['Invoice'] = df['Invoice'].astype('str')
df = df[~df['Invoice'].str.contains('C')]
df

basket = (df[df['Country']== "France"]
        .groupby(['Invoice', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('Invoice'))

basket

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules[ (rules['lift'] >=6) & 
       (rules['confidence'] >= 0.8)]