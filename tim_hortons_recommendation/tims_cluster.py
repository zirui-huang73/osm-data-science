#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
from math import cos, asin, sqrt, pi
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[2]:

# choice = 8 -> TH at Gas station
# choice = 29 -> TH at Metrotown 
choice = 8 #must less than or equal to 112 since there are 122 data 
assert (choice <= 112) & (choice >= 0)
tims = pd.read_csv('./tims/TimHortons.csv')
osm = pd.read_json('./tims/amenities-vancouver.json.gz',lines=True)


# In[3]:


# Drop the rows without name value and get the rows with 'Tim Hortons' osm is the df for tim hortons in osm dataset
osm = osm.dropna(subset=['name'])
osm.reset_index(drop=True, inplace=True)
# save a copy of osm data
original = osm
original.reset_index(drop=True, inplace=True)
osm = osm[osm['name'].str.match('Tim Hortons')]
osm.reset_index(drop=True, inplace=True)


# In[4]:


osm.shape


# In[5]:


tims.shape


# In[6]:


#get the Tim Hortons in Vancourver only 
tims = tims[(tims['longitude'] > -123.5) & (tims['longitude'] < -122) ]
tims = tims[(tims['latitude'] > 49) & (tims['latitude'] < 49.5)]


# In[7]:


# reduce from 4346 to 195
tims.shape


# ## Function Declarations

# In[8]:


def haversine_dist(lat1, lon1, lat2, lon2):
    #adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula?page=1&tab=votes#tab-top
    if((lat2 == 0) and (lon2 == 0 )): 
        return float(0)
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a))*1000.0


# In[9]:


def distance(df,tims):
    dist_calc = np.vectorize(haversine_dist, otypes=[np.float])
    res = dist_calc(df['lat'],df['lon'],tims['latitude'],tims['longitude'])
#it is unlikely there are two TimHortons within 200m
    if np.amin(res) > 200:
        return -1
    else:
        return res.argmin() 


# In[10]:


def parkingTypeProcessing(df, shape):
    res = []
    for i in range(shape):
        if df.iloc[i] == 'Shared':
            res.append(1)
        elif df.iloc[i] == 'Dedicated':
            res.append(2)
        elif df.iloc[i] == 'Street':
            res.append(3)
        else:
            res.append(0)
    return np.array(res)


# In[11]:


def handleMisspell(df, word, correctWord):
    res = []
    for i in range(df.shape[0]):
        if df.iloc[i] == word:
            res.append(correctWord)
        else:
            res.append(df.iloc[i])
    return np.array(res)


# ## Data cleaning

# In[12]:


def tim_dist(x):
    return distance(x,tims)


# In[13]:


osm['tim_index'] = osm.apply(tim_dist,axis=1)


# In[14]:


osm.head()


# In[15]:


# Delete the rows which osm does to have stores correspond to the Tim Hortons dataset
osm = osm[osm['tim_index'] != -1]


# In[16]:


osm.shape


# In[17]:


osm = osm.drop_duplicates(['tim_index'])


# In[18]:


osm.shape


# In[19]:


tims = tims.drop(['deliveryHours', 'franchiseGroupName', 
                  'physicalAddress', 'playgroundType',
                  'pos', 'posRestaurantId','hasBurgersForBreakfast',
                  'hasCatering','hasDriveThru','hasParking','hasPlayground',
                  'diningRoomHours','curbsideHours','drinkStationType','driveThruHours','restaurantPosData'], axis=1)


# In[20]:


# avoid the name collision in the OSM dataset
tims = tims.rename(columns={'name': 'addr'}) 


# In[21]:


tims['tim_index'] = np.arange(tims.shape[0])


# In[22]:


# merge the osm dataset and tims dataset
merge_df = osm.merge(tims, left_on='tim_index', right_on='tim_index')


# In[23]:


# encode the string categories into numerical
le = preprocessing.LabelEncoder()


# In[24]:


merge_df['parkingType'] = parkingTypeProcessing(merge_df['parkingType'], merge_df['parkingType'].shape[0])


# In[25]:


merge_df.head()


# ## Convert the categorical variables to numberical 

# In[26]:


# convert amenity types 0 - cafe, 1 - fast_food
merge_df['amenity']=le.fit_transform(merge_df['amenity'])
assert np.unique(merge_df['amenity']).size == 2


# In[27]:


# convert hasTakeOut 1 - True, 0 - False
merge_df['hasTakeOut']=le.fit_transform(merge_df['hasTakeOut'])
assert np.unique(merge_df['hasTakeOut']).size == 2


# In[28]:


# convert driveThruLaneType 0 - DT, 1 - DT Only, 2 - No DT
merge_df['driveThruLaneType'] = handleMisspell(merge_df['driveThruLaneType'], 'DDT','DT')
merge_df['driveThruLaneType']=le.fit_transform(merge_df['driveThruLaneType'])
assert np.unique(merge_df['driveThruLaneType']).size == 3


# In[29]:


# convert frontCounterClosed 1 - True, 0 - False
merge_df['frontCounterClosed']=le.fit_transform(merge_df['frontCounterClosed'])
assert np.unique(merge_df['frontCounterClosed']).size == 2


# In[30]:


# convert hasBreakfast 1 - True, 0 - False
merge_df['hasBreakfast']=le.fit_transform(merge_df['hasBreakfast'])
assert np.unique(merge_df['hasBreakfast']).size == 2


# In[31]:


# convert hasCurbside 1 - True, 0 - False
merge_df['hasCurbside']=le.fit_transform(merge_df['hasCurbside'])
assert np.unique(merge_df['hasCurbside']).size == 2


# In[32]:


# convert hasDineIn 1 - True, 0 - False
merge_df['hasDineIn']=le.fit_transform(merge_df['hasDineIn'])
assert np.unique(merge_df['hasDineIn']).size == 2


# In[33]:


# convert hasDelivery 1 - True, 0 - False
merge_df['hasDelivery']=le.fit_transform(merge_df['hasDelivery'])
assert np.unique(merge_df['hasDelivery']).size == 2


# In[34]:


# convert hasMobileOrdering 1 - True, 0 - False
merge_df['hasMobileOrdering']=le.fit_transform(merge_df['hasMobileOrdering'])
assert np.unique(merge_df['hasMobileOrdering']).size == 2


# In[35]:


# convert hasTakeOut 1 - True, 0 - False
merge_df['hasTakeOut']=le.fit_transform(merge_df['hasTakeOut'])
assert np.unique(merge_df['hasTakeOut']).size == 2


# In[36]:


# convert hasWifi 1 - True, 0 - False
merge_df['hasWifi']=le.fit_transform(merge_df['hasWifi'])
assert np.unique(merge_df['hasWifi']).size == 2


# In[37]:


# convert mobileOrderingStatus 0 - alpha, 1 - live
merge_df['mobileOrderingStatus']=le.fit_transform(merge_df['mobileOrderingStatus'])
assert np.unique(merge_df['mobileOrderingStatus']).size == 2


# In[38]:


# Use len(tags) --> more tags signify more popular
merge_df['tags_len']= merge_df['tags'].apply(lambda tag: len(tag))


# In[39]:


merge_df.to_csv('merge_tims.csv')


# In[40]:


merge_df.shape


# In[41]:


def closestAmenity(lat1,lon1, osm):
#     lat1, lon1 = tims['latitude'], tims['longitude']
    p = pi/180
    a = 0.5 - np.cos((osm['lat']-lat1)*p)/2 + np.cos(lat1*p) * np.cos(osm['lat']*p) * (1-np.cos((osm['lon']-lon1)*p))/2
    res = pd.DataFrame(columns=['dist','index'])
    res['index'] = np.arange(original.shape[0])
    res['dist'] = 12742 * np.arcsin(np.sqrt(a))*1000
    return res[(res['dist'] > 0)&(res['dist'] < 20)]


# In[42]:


def topThreeAmenity(df):
#     df = pd.DataFrame([[df['lat'],df['lon']]],columns=['latitude','longitude'])
    closest = closestAmenity(df['lat'],df['lon'], original)
    closest = original.iloc[closest['index']].copy()
    if len(closest) == 0:
        return np.array([-1] * 3)
    closest['tag_len'] = closest['tags'].apply(lambda tag: len(tag))
    # get the top 3 popular amenity nearby
    top_three = closest.nlargest(3,'tag_len')['amenity'] 
    # encoder to encode the amenity categories
    le = preprocessing.LabelEncoder()
    amenity_encoder = le.fit(original['amenity'])
    # if there are less than 3 amenity, append the -1 at the end
    padding = np.array([-1] * (3 - len(top_three)))
    res = amenity_encoder.transform(list(top_three))
    res = np.append(res,padding)
    return res


# In[43]:


merge_df['new'] = merge_df.apply(topThreeAmenity,axis=1)


# In[44]:


def splitTopThree(new):
    topThree = list(new)
    l1,l2,l3 = topThree[0],topThree[1],topThree[2]
    df = pd.DataFrame([[l1,l2,l3]],columns=['l1','l2','l3'])
    return df


# In[45]:


merge_df['new'].apply(splitTopThree).head()
merge_split = merge_df['new'].apply(pd.Series)
merge_split = merge_split.rename(columns = lambda x : 'amenity' + str(x))
merge_df = pd.concat([merge_df[:], merge_split[:]], axis=1)


# In[46]:


merge_df.head()


# In[47]:


# Drop unnecessary rows ex. phone number is not useful for the prediction
test_df = merge_df.drop(['lat','lon','name','tags','tim_index','timestamp','latitude',
                         'longitude','phoneNumber','Unnamed: 34','addr','new','franchiseGroupId'],axis=1)


# In[48]:


# extract the ID from the restarnt id
test_df['_id'] = test_df['_id'].str.extract('(\d+)')


# In[49]:


test_df.head()


# In[50]:


def get_pca(X):
    flatten_model = make_pipeline(
        StandardScaler(),
        PCA(5)
    )
    X2 = flatten_model.fit_transform(X)
    return X2


# In[51]:


def get_clusters(X):
    model = make_pipeline(
        KMeans(n_clusters=24)
    )
    model.fit(X)
    return model.predict(X)


# In[52]:


knn = NearestNeighbors(metric = 'manhattan')
test_df = get_pca(test_df)
knn.fit(test_df)


# In[53]:


distances, indices = knn.kneighbors(test_df[choice].reshape(1, -1), n_neighbors = 5)


# In[54]:


def amenityPrint(x):
    x = int(x)
    if x == -1:
        return "empty"
    else:
        le = preprocessing.LabelEncoder()
        le = le.fit(original['amenity'])
        return le.inverse_transform([x])[0]


# In[55]:


merge_df['amenity0'] = merge_df['amenity0'].apply(amenityPrint)
merge_df['amenity1'] = merge_df['amenity1'].apply(amenityPrint)
merge_df['amenity2'] = merge_df['amenity2'].apply(amenityPrint)


# In[56]:


# it will output the 4 closest tims horton
knn_data = []
print("KNN result with address: ")
for i in range(0, len(distances.flatten())):
    if i == 0:
        print(merge_df.iloc[choice]['addr'])
        print(merge_df.iloc[choice][['lat','lon']])
        knn_data.append(merge_df.iloc[choice].values)
    else:
        print("++++++++++++++Neighbors++++++++++++++++++")
        print(merge_df.iloc[indices.flatten()[i]]['addr'])
        print(merge_df.iloc[indices.flatten()[i]][['lat','lon']])
        knn_data.append(merge_df.iloc[indices.flatten()[i]].values)


# In[57]:


knn_df = pd.DataFrame(knn_data)


# In[58]:


knn_df.to_csv('knn_result.csv',sep=',',header=merge_df.columns)


# In[59]:


# apply kmean here to see if it get a simialr result as knn
km = KMeans(n_clusters=24).fit(test_df)
cluster_map = pd.DataFrame()
cluster_map['data_index'] = np.arange(112)
cluster_map['cluster'] = km.labels_


# In[60]:


kmean_df = cluster_map[cluster_map.cluster == (cluster_map.iloc[choice]['cluster'])]
kmean_data = []
for i in range(kmean_df.shape[0]):
    kmean_data.append(merge_df.iloc[kmean_df['data_index'].iloc[i]].values)
kmean = pd.DataFrame(kmean_data)
kmean.to_csv('kmean_result.csv',sep=',',header=merge_df.columns)


# In[61]:


print("The KMEAN result with address: ")
for i in range(kmean_df.shape[0]):
    index = kmean_df.iloc[i]['data_index']
    print(merge_df.iloc[index]['addr'])
    print(merge_df.iloc[index][['lat','lon']])
    print(" ")


# In[62]:


print("The KMEAN result(in index): ")
print(cluster_map[cluster_map.cluster == (cluster_map.iloc[choice]['cluster'])])


# In[63]:


print("The Knn result(in index): "+ str(indices.flatten()))


# In[ ]:




