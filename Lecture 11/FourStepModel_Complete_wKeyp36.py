#!/usr/bin/env python
# coding: utf-8

# ## Four-Step Model in Python by David Bailey (First Part)

#  source: https://davidabailey.com/articles/Creating-a-Four-step-Transportation-Model-in-Python 
#  This codes reads Travels produced by cars from the Census Web page and reads Employment from the 
#  Bureau of Labor statistics, generates Trip Production Trip Attraction per County and a Trip
#  Distribution Model, we show today the first part.

# In[2]:


get_ipython().system('pip install haversine')


# In[491]:


get_ipython().system('pip install ipfn')


# In[1]:


import requests
import pandas
import geopandas
import json
import math
from haversine import haversine
import ipfn
import networkx
from matplotlib import pyplot
from matplotlib import patheffects
get_ipython().run_line_magic('matplotlib', 'inline')


# In[493]:


#conda install -c conda-forge libspatialindex


# In[2]:


import sys
print("The Python version is %s.%s.%s" % sys.version_info[:3])


# ## Check:
# https://conda.io/docs/user-guide/tasks/manage-environments.html#activate-env
# 
# http://geopandas.org/install.html

# In[3]:


url = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/37/query?where=state%3D06&f=geojson'
r = requests.get(url)
zones = geopandas.GeoDataFrame.from_features(r.json()['features'])


# In[4]:


centroidFunction = lambda row: (row['geometry'].centroid.y, row['geometry'].centroid.x)
zones['centroid'] = zones.apply(centroidFunction, axis=1)


# In[5]:


zones.plot()
pyplot.show()


# In[6]:


gdf = geopandas.GeoDataFrame


# In[7]:


zones.dtypes


# Query example from the US Census API:
# https://www.census.gov/data/developers/guidance/api-user-guide/query-examples.html

# In[ ]:


#needs Census api key
url ='https://api.census.gov/data/2015/acs/acs5?get=NAME,B08015_001E&for=county&in=state:06&key='
r = requests.get(url) 


# In[9]:


r.json()


# In[10]:


#Production = pandas.DataFrame(r.json()[1:], columns = r.json()[0])
Production = pandas.DataFrame(r.json()[1:], columns = r.json()[0], dtype='int')
nameSplit = lambda x: x.split(',')[0]
Production['NAME'] = Production['NAME'].apply(nameSplit)
zones = pandas.merge(zones, Production)
zones['Production'] = zones['B08015_001E']


# In[11]:


zones.head()


# In[12]:


zones


# In[13]:


zones.dtypes


# In[14]:


def getEmployment(state, county):
    prefix = 'EN'
    seasonal_adjustment = 'U'
    area = format(state, "02d") + format(county, "03d")
    data_type = '1'
    size = '0'
    ownership = '0'
    industry = '10'
    seriesid = prefix + seasonal_adjustment + area + data_type + size + ownership + industry
    headers = {'Content-type': 'application/json'}
    #needs LBS api key
    data = json.dumps({"seriesid": [seriesid],"startyear":"2015", "endyear":"2015","registrationKey":""})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    employment = p.json()['Results']['series'][0]['data'][0]['value']
    return(employment)


# Read me Bureau of labor statistics: https://www.bls.gov/help/hlpforma.html
#         
# LBS API example: https://www.bls.gov/developers/api_python.htm#python1
#         
# Get your API: https://data.bls.gov/registrationEngine/

# In[15]:


#employment = lambda row: int(getEmployment('06','003'))
employment = lambda row: int(getEmployment(row['state'],row['county']))


# In[19]:


get_ipython().system('pip install apply')


# This part takes 1 min for each of the 58 CA counties it prints the area code and #employments as it runs

# In[16]:


zones['Attraction'] = None


# ### This part retrieves employment from Census 1sec for each county

# In[17]:


zones['Attraction'] = zones.transpose().apply(employment)


# In[259]:


zones.head()


# In[260]:


pandas.set_option('display.float_format', lambda x: '%.0f' % x)
zones[['Production', 'Attraction']].head()


# In[261]:


zones.index = zones.NAME #adds column Name to index column and sorts
zones.sort_index(inplace=True)


# In[262]:


pandas.set_option('display.float_format', lambda x: '%.0f' % x)
zones[['Production', 'Attraction']].head()


# ### Here we scale Trip production to match total employment

# In[18]:


zones['Production'] = zones['Production'] * zones.sum()['Attraction'] / zones.sum()['Production'] 


# In[19]:


pandas.set_option('display.float_format', lambda x: '%.0f' % x)
zones[['Production', 'Attraction']].head()


# # This part uses the ipfn function to generate trip distribtion

# In[20]:


#Runs the ipfn method from a dataframe df, aggregates/marginals and the dimension(s) preserved.
#For example:
from ipfn import ipfn
import pandas as pd
age = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
distance = [10,20,30,40,10,20,30,40,10,20,30,40]
m = [8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.]
df = pd.DataFrame()
df['age'] = age
df['distance'] = distance
df['total'] = m
print(df)
xip = df.groupby('age')['total'].sum()


# In[21]:


xip  #by age initially sums 65


# In[22]:


#xip changes to 60
xip.loc[30] = 20
xip.loc[40] = 18
xip.loc[50] = 22


# In[23]:


xpj = df.groupby('distance')['total'].sum()


# In[24]:


xpj #stays in 65


# In[25]:


dimensions = [['age'], ['distance']]
aggregates = [xip, xpj]
IPF = ipfn.ipfn(df, aggregates, dimensions)
df = IPF.iteration()


# In[26]:


print(df)


# In[27]:


print(df.groupby('age')['total'].sum(), xip) #after ipf goes to xip


# In[28]:


print(df.groupby('distance')['total'].sum(), xpj) #without changing totals by distance


# In[34]:


def costFunction(zones, zone1, zone2, beta):
    cost = math.exp(-beta * haversine(zones[zone1]['centroid'], zones[zone2]['centroid']))
    return(cost)


# In[35]:


def costMatrixGenerator(zones, costFunction, beta):
    originList = []
    for originZone in zones:
        destinationList = []
        for destinationZone in zones:
            destinationList.append(costFunction(zones, originZone, destinationZone, beta))
        originList.append(destinationList)
    return(pandas.DataFrame(originList, index=zones.columns, columns=zones.columns))


# In[36]:


costMatrix=[]
beta = 0.01


# In[37]:


costMatrix = costMatrixGenerator(zones.transpose(),costFunction,beta)


# In[38]:


costMatrix


# In[39]:


def tripDistribution(tripGeneration,costMatrix):
    costMatrix['ozone'] = costMatrix.columns
    costMatrix = costMatrix.melt(id_vars=['ozone'])
    costMatrix.columns = ['ozone', 'dzone', 'total']
    production = tripGeneration['Production']
    production.index.name = 'ozone'
    attraction = tripGeneration['Attraction']
    attraction.index.name = 'dzone'
    aggregates = [production, attraction]
    dimensions = [['ozone'], ['dzone']]
    IPF = ipfn.ipfn(costMatrix, aggregates, dimensions)
    trips = IPF.iteration()
    return(trips.pivot(index='ozone', columns='dzone', values='total'))


# In[40]:


trips = tripDistribution(zones, costMatrix)


# In[41]:


def modeChoiceFunction(zones, zone1, zone2, modes):
    distance = haversine(zones[zone1]['centroid'], zones[zone2]['centroid'])
    probability = {}
    total = 0.0
    for mode in modes:
        total = total + math.exp(modes[mode] * distance)
    for mode in modes:
        probability[mode] = math.exp(modes[mode] * distance) / total
    return(probability)


# In[42]:


def probabilityMatrixGenerator (zones, modeChoiceFunction, modes):
    probabilityMatrix = {}
    for mode in modes:
        originList = []
        for originZone in zones:
            destinationList = []
            for destinationZone in zones:
                destinationList.append(modeChoiceFunction(zones, originZone, destinationZone, modes)[mode])
            originList.append(destinationList)
        probabilityMatrix[mode] = pandas.DataFrame(originList, index=zones.columns, columns=zones.columns)
    return(probabilityMatrix)


# In[43]:


modes = {'walking': .05, 'cycling': .05, 'driving': .05}
probabilityMatrix = probabilityMatrixGenerator(zones.transpose(), modeChoiceFunction, modes)


# In[45]:


drivingTrips = trips * probabilityMatrix['driving']


# In[46]:


def routeAssignment(zones, trips):
    G = networkx.Graph()
    G.add_nodes_from(zones.columns)
    for zone1 in zones:
        for zone2 in zones:
            if zones[zone1]['geometry'].touches(zones[zone2]['geometry']):
                G.add_edge(zone1, zone2, distance = haversine(zones[zone1]['centroid'], zones[zone2]['centroid']), volume=0.0)
    for origin in trips:
        for destination in trips:
            path = networkx.shortest_path(G, origin, destination)
        for i in range(len(path) - 1):
            G[path[i]][path[i + 1]]['volume'] = G[path[i]][path[i + 1]]['volume'] + trips[zone1][zone2]
    return(G)


# In[47]:


def visualize(G, zones):
    fig = pyplot.figure(1, figsize=(10, 10), dpi=90)
    ax = fig.add_subplot(111)
    zonesT = zones.transpose()
    for i, row in zones.transpose().iterrows():
        text = pyplot.annotate(s=row['NAME'], xy=((row['centroid'][1], row['centroid'][0])), horizontalalignment='center', fontsize=6)
        text.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white'), patheffects.Normal()])
        
    for zone1,zone2,w in G.edges(data=True):
        volume=list(dict(w).values())
        #print zone1,' ',zone2,' ',volume[0]
        x = [zones[zone1]['centroid'][1], zones[zone2]['centroid'][1]]
        y = [zones[zone1]['centroid'][0], zones[zone2]['centroid'][0]]
        ax.plot(x, y, color='#444444', linewidth=volume[0]/500, solid_capstyle='round', zorder=1)
    zonesT.plot(ax = ax,color='#DAF7A6')
    pyplot.show(block=False)    


# In[48]:


G = routeAssignment(zones.transpose(), drivingTrips)


# In[49]:


G


# In[50]:


visualize(G, zones.transpose())


# ## For playing with parameters

# In[59]:


# Trip Distribution
beta = 0.1
costMatrix = costMatrixGenerator(zones.transpose(), costFunction, beta)
Trips = tripDistribution(zones, costMatrix)
# Mode Choice
modes = {'walking': .05, 'cycling': .05, 'driving': .05}
probabilityMatrix = probabilityMatrixGenerator(zones.transpose(), modeChoiceFunction, modes)
drivingTrips = Trips * probabilityMatrix['driving']
# Route Assignment
G = routeAssignment(zones.transpose(), drivingTrips)
visualize(G, zones.transpose())


# In[54]:


G.number_of_edges()


# In[55]:


58*58


# In[56]:


len(zones)


# In[ ]:




