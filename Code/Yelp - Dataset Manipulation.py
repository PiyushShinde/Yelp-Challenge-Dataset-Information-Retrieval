# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2 15:36:23 2017

@author: Piyush Shinde
"""

import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
    
#Read data
business = pd.read_csv('business.csv', encoding = "latin-1")
review = pd.read_csv('review.csv', encoding = "latin-1")
user = pd.read_csv('user.csv', encoding = "latin-1")

#Select 3 cities with similar number of business reviews
print(business.city.value_counts())

#Selecting cities Charlotte, Scottsdale and Pittsburgh (Number of Reviews - 7557, 7510, 5688)
cities = ['Charlotte','Scottsdale','Pittsburgh']
#Create dictionaries for each city
train_city = {}
val_city = {}
test_city = {}

for i in tqdm(range(3)): 
    business_city = business[business.city == cities[i]]
    bus_rev_city = business_city.merge(review, on = 'business_id')
    bus_rev_user_city = bus_rev_city.merge(user, on='user_id')
    city_df = bus_rev_user_city[['user_id','business_id','categories','stars_y','date','text'
                  ,'useful_x','funny_x','cool_x']]
    city_df = city_df.rename(columns = {'stars_y':'stars_review','useful_x':'useful_review','funny_x':'funny_review', 'cool_x' :'cool_review'})
    city_rest = city_df[city_df.categories.apply(lambda a: 'Restaurants' in a)]
    city_rest = city_rest.reset_index()
    city_rest = city_rest.drop(['index'],axis=1)
    city_rest_users = pd.DataFrame(city_rest.user_id.value_counts())
    city_rest_users = city_rest_users.reset_index()
    city_rest_users = city_rest_users.rename(columns = {"index":"user_id","user_id":"counts"})
    city_rest_users = city_rest_users[city_rest_users.counts >= 20]
    city_final = pd.merge(city_rest,city_rest_users, on=['user_id'], how='inner')

    train_city[i] = city_final[(pd.DatetimeIndex(city_final.date).year<2015)]
    val_city[i] = city_final[(pd.DatetimeIndex(city_final.date).year>=2015) & (pd.DatetimeIndex(city_final.date).year<2016)] 
    test_city[i] = city_final[(pd.DatetimeIndex(city_final.date).year>=2016)]
    train_city[i].to_csv("data/"+str(cities[i])+"_train.csv")
    val_city[i].to_csv("data/"+str(cities[i])+"_val.csv")
    test_city[i].to_csv("data/"+str(cities[i])+"_test.csv")    

#Charlotte = 0, Scottsdale = 1, Pittsburgh = 2
#Read train data
city_train = pd.read_csv("data/"+str(cities[i])+"_train.csv",usecols=range(1,11),encoding='latin-1')

#Read validation data
city_val = pd.read_csv("data/"+str(cities[i])+"_val.csv",usecols=range(1,11),encoding='latin-1')

#Read test data
city_test = pd.read_csv("data/"+str(cities[i])+"_test.csv",usecols=range(1,11),encoding='latin-1')

#----------------------------------------------------------------------------------------------------
"""print(train_city[0].shape)
print(val_city[0].shape)
print(test_city[0].shape)
print(train_city[1].shape)
print(val_city[1].shape)
print(test_city[1].shape)
print(train_city[2].shape)
print(val_city[2].shape)
print(test_city[2].shape)

(23746, 10)
(6136, 10)
(11188, 10)
(18387, 10)
(3707, 10)
(5371, 10)
(18570, 10)
(5665, 10)
(9226, 10)"""
