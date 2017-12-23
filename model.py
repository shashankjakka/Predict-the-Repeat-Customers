
# coding: utf-8

# In[455]:


## Import the required Libraries
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine


# In[2]:


## Read excels
customers=pd.read_excel('/Users/shashank/Downloads/Assignment_AI/Assignment1of2/Data/LU_CUSTOMER.xlsx')
items=pd.read_excel('/Users/shashank/Downloads/Assignment_AI/Assignment1of2/Data/LU_ITEM.xlsx')
orders=pd.read_excel('/Users/shashank/Downloads/Assignment_AI/Assignment1of2/Data/LU_ORDER.xlsx')
order_details=pd.read_excel('/Users/shashank/Downloads/Assignment_AI/Assignment1of2/Data/ORDER_DETAIL.xlsx')


# In[406]:


pd.set_option('max_rows',50)
pd.set_option('max_columns',25)


# In[4]:


print customers.shape,items.shape,orders.shape,order_details.shape


# In[156]:


## Merge customers and order_details
df1=customers[['CUSTOMER_ID','BRACKET_DESC','AGE']] ## Though I don't think Age is reveleant here
df=(pd.merge(order_details, df1, on='CUSTOMER_ID'))


# In[158]:


## Get Categories for the item_ids from item detail
df2=items[['ITEM_ID','SUBCAT_ID']]
df=pd.merge(df,df2,on='ITEM_ID')


# In[159]:


### EMP_ID-- No description given
### PROMOTION_ID,DISCOUNT- If these weren't a constant could have been very handy

df.drop(['ORDER_ID','EMP_ID','PROMOTION_ID','QTY_SOLD','DISCOUNT'],axis=1,inplace=True)


# In[160]:


# use date as index for easy slicing
t=df.set_index(df['ORDER_DATE'].sort_values())


# In[161]:


t.head()


# In[163]:


## lets train till the end of October and validate our results on the month of November

test=t['2004-11-01':'2004-11-30']
t=t[:'2004-10-31']


# In[164]:


# past 1,2,3 and 6 months data excluding 2004 December
months1=t['2004-10-1':'2004-10-31']
months2=t['2004-09-1':'2004-10-31']
months3=t['2004-08-1':'2004-10-31']
months6=t['2004-05-1':'2004-10-31']


# In[165]:


months1.tail()


# In[166]:


## frequency of Customers in last 1,2,3 and 6 months
freq1={}
for i in customers.CUSTOMER_ID.unique():
    freq1[i]=len(months1[months1['CUSTOMER_ID']==i])
    
freq2={}
for i in customers.CUSTOMER_ID.unique():
    freq2[i]=len(months2[months2['CUSTOMER_ID']==i])
    
freq3={}
for i in customers.CUSTOMER_ID.unique():
    freq3[i]=len(months3[months3['CUSTOMER_ID']==i])
    
freq6={}
for i in customers.CUSTOMER_ID.unique():
    freq6[i]=len(months6[months6['CUSTOMER_ID']==i])


# In[186]:


# total frequency

freq={}
for i in customers.CUSTOMER_ID.unique():
    freq[i]=len(t[t['CUSTOMER_ID']==i])


# In[167]:


## Money Spent in the last 1,2,3 and 6 months
spends1={}
for i in customers.CUSTOMER_ID.unique():
    spends1[i]=int(months1[months1['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())
    
spends2={}
for i in customers.CUSTOMER_ID.unique():
    spends2[i]=int(months2[months2['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())
    
spends3={}
for i in customers.CUSTOMER_ID.unique():
    spends3[i]=int(months3[months3['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())

spends6={}
for i in customers.CUSTOMER_ID.unique():
    spends6[i]=int(months6[months6['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())


# In[168]:


# total money spennt

spendsTotal={}
for i in customers.CUSTOMER_ID.unique():
    spendsTotal[i]=int(t[t['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())


# In[169]:


train=pd.DataFrame()


# In[170]:


## Add a feature which mesure the the months since the last visit of each customer

last_occurances=t.drop_duplicates(subset='CUSTOMER_ID',keep='last')


# In[171]:


last_occurances.head()


# In[172]:


last_visit={}
for ind,row in last_occurances.iterrows():
    last_visit[row['CUSTOMER_ID']]=(12-(ind.month+1))


# In[173]:


last_occurances[last_occurances["CUSTOMER_ID"]==1]


# In[187]:


## Build a data set with all the extracted features
train['CUSTOMER_ID']=pd.Series(customers.CUSTOMER_ID.unique())


# In[188]:


train['TOTAL_SPENDS']=pd.Series(spendsTotal.values())
train['PAST_1MONTH_SPENDS']=pd.Series(spends1.values())
train['PAST_2MONTH_SPENDS']=pd.Series(spends2.values())
train['PAST_3MONTH_SPENDS']=pd.Series(spends3.values())
train['PAST_6MONTH_SPENDS']=pd.Series(spends6.values())


# In[189]:


train['TOTAL_FREQ']=pd.Series(freq.values())
train['PAST_1MONTH_FREQ']=pd.Series(freq1.values())
train['PAST_2MONTH_FREQ']=pd.Series(freq2.values())
train['PAST_3MONTH_FREQ']=pd.Series(freq3.values())


# In[190]:


train['MONTHS_SINCE_LAST_VISIT']=pd.Series(last_visit.values())


# In[195]:


train.head()


# In[192]:


## regex to extract numbers from income bracket and average them  as income
## Maybe better to use continuos income instead of a categorial variable for Income

customers['INCOME']=customers['BRACKET_DESC'].apply(lambda x:np.array(map(int, re.findall(r'\d+', x))).mean())
cust=customers[['CUSTOMER_ID','INCOME']]
train=pd.merge(train,cust,on='CUSTOMER_ID')


# In[193]:


ids=train['CUSTOMER_ID']


# In[210]:


## Scale for KMeans
sc=StandardScaler()
X=sc.fit_transform(train.iloc[:,1:].values)


# In[529]:


km=KMeans(n_clusters=3)
km.fit(X)


# In[530]:


preds=km.predict(X)


# In[531]:


train["CLUSTER"]=pd.Series(preds)


# In[583]:


train.CLUSTER.value_counts(normalize=True)


# In[534]:


train[train['CLUSTER']==0].describe()


# In[535]:


train[train['CLUSTER']==1].describe()


# In[536]:


train[train['CLUSTER']==2].describe()


# In[537]:


##People who actually visisted in Novemeber
test.CUSTOMER_ID.unique()


# In[539]:


## People who we predicted to visit again
visited=(train['CLUSTER']==2)|(train['CLUSTER']==1)
predicted_visited=np.array(ids[visited])


# In[540]:


len(test.CUSTOMER_ID.unique()),len(predicted_visited)


# In[549]:


imp=pd.DataFrame({'user':predicted_visited})


# In[550]:


imp.head()


# In[541]:


len(np.intersect1d(test.CUSTOMER_ID.unique(),predicted_visited))


# In[ ]:


## Predicted true for 4932 Customers out of 8870
## Remember these predictions are for every month not just for a specific month of November
## As it was only asked to predict the most probable customers who will visit again


# In[542]:


len(predicted_visited)


# In[155]:


test.drop_duplicates(subset=['CUSTOMER_ID'])


# In[220]:


ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)


# In[221]:


preds=ward.labels_


# In[222]:


preds


# In[562]:


pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)


# In[563]:


fig=pd.DataFrame({'X':X_pca[:,0],'Y':X_pca[:,1],'Labels':train.iloc[:,-1]})


# In[582]:


flatui = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]

sns.lmplot(x='X',y='Y',hue="Labels",data=fig,fit_reg=False,palette=flatui)


# In[250]:


pd.Series(preds).value_counts(normalize=True)


# In[260]:


train[train['CLUSTER']==4].describe()


# In[255]:


train.head()


# In[254]:


train.isnull().sum()


# In[279]:


len(order_details.CUSTOMER_ID.unique())


# In[560]:


## Recommendation Starts


# In[288]:


## Considering a subset of data
rec=t[:'2004-01-01']


# In[295]:


rec=rec.reset_index(drop=True)


# In[307]:


data=rec.drop_duplicates(subset=['CUSTOMER_ID','ITEM_ID'])


# In[349]:


data=data[['CUSTOMER_ID','ITEM_ID']]


# In[350]:


data.head()


# In[362]:


data=pd.merge(data,items[['ITEM_ID','ITEM_NAME']],on='ITEM_ID')


# In[363]:


data.drop(['ITEM_ID'],axis=1,inplace=True)


# In[364]:


data


# In[366]:


data.ITEM_NAME.nunique()


# In[376]:


products=list(data.ITEM_NAME.unique())


# In[374]:


data['CUSTOMER_ID'].nunique()


# In[439]:


li=[]
for i in range(1,10001):
    to_append=[]
    user_list=list(data[data['CUSTOMER_ID']==i]['ITEM_NAME'])
    for i in products:
        if i in user_list:
            to_append.append(1)
        else:
            to_append.append(0)
    li.append(to_append)


# In[480]:


matrix=pd.DataFrame()


# In[481]:


matrix=pd.DataFrame(li,columns=products)


# In[482]:


matrix.insert(0,'User',pd.Series(range(0,10000)))


# In[465]:


data_ibs = pd.DataFrame(index=matrix.drop(['User'],axis=1).columns,columns=matrix.drop(['User'],axis=1).columns)


# In[466]:


for i in range(0,len(data_ibs.columns)) :
    for j in range(0,len(data_ibs.columns)) :
        data_ibs.iloc[i,j] = 1-cosine(matrix.drop(['User'],axis=1).iloc[:,i],matrix.drop(['User'],axis=1).iloc[:,j])


# In[468]:


data_ibs.head(3)


# In[467]:


data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,11))
 
for i in range(0,len(data_ibs.columns)):
    data_neighbours.iloc[i,:10] = data_ibs.iloc[0:,i].sort_values(ascending=False)[:10].index


# In[489]:


data_neighbours.head(3)


# In[508]:


data_neighbours.head(6).iloc[:6,:]


# In[511]:


data_neighbours.loc[product][:10]


# In[483]:


def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)


# In[484]:


data_sims = pd.DataFrame(index=matrix.index,columns=matrix.columns)
data_sims.iloc[:,:1] = matrix.iloc[:,:1]


# In[485]:


data_sims.head(3)


# In[478]:


data_sims.index[0]


# In[520]:


for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        print i,j
        user = data_sims.index[i]
#         product = str(data_sims.columns[j])
        if matrix.iloc[i][j] == 1:
            data_sims.iloc[i][j] = 0
        else:
            product_top_names = data_neighbours.loc[str(data_sims.columns[j])][:10]
            product_top_sims = data_ibs.loc[product].sort_values(ascending=False)[:10]
            user_purchases = matrix.drop(['User'],axis=1).loc[user,product_top_names]
 
            data_sims.iloc[i][j] = getScore(user_purchases,product_top_sims)


# In[523]:


data_recommend = pd.DataFrame(index=data_sims.index, columns=['user','1','2','3','4','5','6'])
data_recommend.iloc[0:,0] = data_sims.iloc[:,0]


# In[524]:


for i in range(0,len(data_sims.index)):
    data_recommend.iloc[i,1:] = data_sims.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()
 


# In[546]:


data_recommend.head()


# In[552]:


return_customer=pd.merge(imp,data_recommend,on='user')


# In[558]:


return_customer.to_csv('/Users/shashank/Desktop/ReturnCustomer&Recommendations.csv'
                       ,index=False)


# In[559]:


return_customer

