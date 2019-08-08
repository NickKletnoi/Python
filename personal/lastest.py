
# coding: utf-8

# In[4]:


import pyodbc
conn = pyodbc.connect('Driver={SQL Server};Server=DESKTOP-C949P6F;Database=Voyager;Trusted_Connection=yes;')
rs = conn.cursor()
rs.execute('select * from dw.AccountDimTest')

for row in rs:
    print(row)

for i in range(2):
    print('hello')


# In[5]:


from turtle import *
shape('turtle')
for i in range(4):
    forward(100)
    right(90)


# In[4]:


'first degree equation'
def equation(a,b,c,d):
    return (d-b)/(a-c)

equation(.5,.66,.25,.87)


# In[12]:


from math import sqrt
'quadratic equation'
def quad(a,b,c):
    x1=(-b + sqrt(b**2 - 4*a*c))/(2*a)
    x2=(-b - sqrt(b**2 - 4*a*c))/(2*a)
    return(x1,x2)
quad(2,7,-15)


# In[ ]:



   


# In[3]:


def apply_discount(product, discount):
    price = int(product['price']) * (1.0 - discount)
    assert 0 <= price <= product['price']
    return price

shoes = {'name':'Fancy Shoes','price':14900}

apply_discount(shoes,0.25)
    
    


# In[134]:


import numpy as np
import pandas as pd
import re

df = pd.read_table('C:/temp/test.txt', delim_whitespace=True, names=('year', 'pop','name'),
                   dtype={'year': np.int, 'pop': np.int,'name':np.str})

#print(df)
#print(df[['year','name']])
#print(df[['year','name','pop']])
#print(df[['name']])
#print(df.loc[:13])   
#------------------------------
#print(df.groupby('year')['pop'].mean())
#fx = df.groupby('year')['pop'].mean()
#fx.plot()
#----------------------------------
# the concept of a dataframe is the  = of a dict
# which is the equivalent of the following:
# date conversion: born_dt = pd.to_datetime(mypeeps['born'],format='%Y-%m-%d')

years = [1957,1934]
names = ['Fred','Mike']

mypeeps = pd.DataFrame(
        data={'year':[1957,1934,1967],
               'pop':[3456,6765,9878],
                'name':['Fred','Mike','Sam']})

mypeeps_f=mypeeps[mypeeps.name.isin(names) & mypeeps.year.isin(years)]
mypeeps_f1=mypeeps_f[(mypeeps_f['pop'] > 5000)]


#print(mypeeps_f1[['year','name','pop']])

#-------------joins in Python---------------------------------------
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
#------------------------------------------------------------------

df40 = pd.merge(df1, df2, on='employee', how='inner')
df50 = pd.merge(df40,df3, on='group', how='inner')
df50[['group','supervisor']]
#------------------------------------ Regular Expression Matching ---
df51=df50[df50.supervisor.str.contains(r'St')]
#http://www.regexlab.com/wild2regex

print(df51)

#----------  filter a groupby the equivalent of having in sql 
def filter1(x):
    return x['hire_date'].sum() > 2012


# Define the aggregation calculations
aggregations = {
#     'duration': { # work on the "duration" column
#         'total_duration': 'sum',  # get the sum, and call this result 'total_duration'
#         'average_duration': 'mean', # get mean, call result 'average_duration'
#         'num_calls': 'count'
#     },
#     'date': {     # Now work on the "date" column
#         'max_date': 'max',   # Find the max, call the result "max_date"
#         'min_date': 'min',
#         'num_days': lambda x: max(x) - min(x)  # Calculate the date range per group
#     },
    'hire_date': ["count", "max","sum"]  # Calculate two results for the 'network' column with a list
}

# Perform groupby aggregation by "month", but only on the rows that are of type "call"
# data[data['item'] == 'call'].groupby('month').agg(aggregations)


# df60=df50.groupby('group').sum()
# df70 =df50.groupby('group').filter(filter1) 
# df80=df50.groupby(['group', 'supervisor'])['hire_date'].sum()
df80=df50.groupby(['group', 'supervisor']).agg(aggregations)
#df90=df80[df80.group.str.contains(r'St')]
#df70['EmployeeGroup'] = df70.employee + df70.group


# print(df50)
# print(df60)
# print(df70)
print(df80)
#print(mypeeps_f1)

#-----------------------------------------------------------------

#decade = 10 * (planets['year'] // 10)
#decade = decade.astype(str) + 's'
#decade.name = 'decade'
#planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

#df90 = df80.groupby(['group','supervisor'], as_index=False).sum()

#Print(df90)



                

