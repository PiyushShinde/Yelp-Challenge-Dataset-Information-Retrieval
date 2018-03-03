# -*- coding: utf-8 -*-
"""

@author: saurabh
"""

import pandas as pd
import numpy as np
import warnings
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
import datetime as date
#from tqdm import tqdm

warnings.filterwarnings("ignore")

#Selecting cities Charlotte, Scottsdale and Pittsburgh (Number of Reviews - 7557, 7510, 5688)
cities = ['Charlotte','Scottsdale','Pittsburgh']

#Charlotte = 0, Scottsdale = 1, Pittsburgh = 2
#i = 0,1,2
#Read train data
i = 0
city_train = pd.read_csv("data/"+str(cities[i])+"_train.csv",usecols=range(1,11),encoding='latin-1')

#Read validation data
city_val = pd.read_csv("data/"+str(cities[i])+"_val.csv",usecols=range(1,11),encoding='latin-1')

#Read test data
city_test = pd.read_csv("data/"+str(cities[i])+"_test.csv",usecols=range(1,11),encoding='latin-1')

# Dropping Duplicates
city_train.drop_duplicates(subset = ['categories'],inplace=True)
city_val.drop_duplicates(subset = ['categories'],inplace=True)
city_test.drop_duplicates(subset = ['categories'],inplace=True)

#To select test set with user and business ids in city_test only if they are present in city_train
#Same function is going to be used to get validation set 
def get_test(city_train,city_test):
    #Select ids in test if present in train
    user_ids = [id for id in city_test.user_id.unique() if id in city_train.user_id.unique()]
    business_ids = [id for id in city_test.business_id.unique() if id in city_train.business_id.unique()]
    final_test = city_test[city_test.user_id.apply(lambda a: a in user_ids)].copy()
    final_test = city_test[city_test.user_id.apply(lambda a: a in user_ids)].copy()
    final_test = final_test[final_test.business_id.apply(lambda a: a in business_ids)].copy()
    final_test = final_test.reset_index()
    final_test = final_test.drop('index', axis = 1)
    return final_test

def get_user_rev_train(city_train):
    user_rev_train = city_train[['user_id','categories']].groupby('user_id')['categories'].apply(list).reset_index()
    
    ## Updated
    #user_rev_train.text = user_rev_train.text.apply(lambda a: "".join(re.sub(r'[^\w\s]',' ',str(a))).replace("\n"," "))
    #user_rev_train.categories = ''.join(user_rev_train.categories)
    user_rev_train.categories = [''.join(item) for item in user_rev_train.categories]
    return user_rev_train

#Has all users from city_train with their corresponding text grouped as a list
user_reviews_train = get_user_rev_train(city_train)

###### Snippet for TF-IDF
#TF-IDF
vectorizer_rest_rev_train = TfidfVectorizer(binary=False
                             , stop_words = 'english'
                             , min_df = 5, max_df=.8)

vectorizer_rest_rev_train = vectorizer_rest_rev_train.fit(user_reviews_train.categories.apply(lambda a:a.lower()))

tfidf = vectorizer_rest_rev_train.transform(user_reviews_train.categories.apply(lambda a:a.lower()))
#####

#***********************Common function definitions*****************************
def calc_sim_user_rating_WA(target_user_id,target_business_id):
    test_rest_user_id = city_train[city_train.business_id == target_business_id].copy()
    user_reviews_train = get_user_rev_train(city_train)
    target_user_vector = tfidf[user_reviews_train[user_reviews_train.user_id == target_user_id].index[0]].copy()
    dist_user = []
    for user_id in test_rest_user_id.user_id.values:
        dist_user.append(cosine_similarity(tfidf[user_reviews_train[user_reviews_train.user_id==user_id].index[0]],
                                           target_user_vector)[0][0])
        
    test_rest_user_id['similarity'] = dist_user
    #Filtering technique = Weighted Average
    weighted_avg = []
    for user_id in test_rest_user_id.user_id.values:
        test_user_similarity = test_rest_user_id.similarity[test_rest_user_id.user_id==user_id].values[0]
        rating_diff = test_rest_user_id.stars_review[test_rest_user_id.user_id==user_id].values[0]
        -np.average(city_train[city_train.user_id ==user_id].stars_review.values)
        weighted_avg.append(np.dot(test_user_similarity,rating_diff)/np.sum(test_rest_user_id.similarity[test_rest_user_id.user_id==user_id].values))
    test_rest_user_id['weighted_avg'] = weighted_avg
        
    return np.average(test_rest_user_id.weighted_avg)

def calc_sim_user_rating(target_user_id,target_business_id):
    test_rest_user_id = city_train[city_train.business_id == target_business_id].copy()
    user_reviews_train = get_user_rev_train(city_train)
    target_user_vector = tfidf[user_reviews_train[user_reviews_train.user_id == target_user_id].index[0]].copy()
    dist_user = []
    for user_id in test_rest_user_id.user_id.values:
        dist_user.append(cosine_similarity(tfidf[user_reviews_train[user_reviews_train.user_id==user_id].index[0]],
                                           target_user_vector)[0][0])
        
    test_rest_user_id['similarity'] = dist_user    
    #Filtering technique = Calc percentile and filter percentiles > 80%
    test_rest_user_id = test_rest_user_id[test_rest_user_id.rank(pct=True).similarity > .8]
     
    rating_diff = []
    for user_id in test_rest_user_id.user_id.values:
        rating_diff.append(test_rest_user_id.stars_review[test_rest_user_id.user_id==user_id].values[0]
                            -np.average(city_train[city_train.user_id ==user_id].stars_review.values))
    test_rest_user_id['rating_diff'] = rating_diff
    
    return np.average(test_rest_user_id.rating_diff)

def calc_rating(sampled_test):
    pred_ratings = []
    for id_ in sampled_test.index:
        user_id = sampled_test[sampled_test.index==id_].user_id.values[0]
        business_id = sampled_test[sampled_test.index==id_].business_id.values[0]
        try:
            #user_rating_shift = calc_sim_user_rating_WA(user_id,business_id)
            user_rating_shift = calc_sim_user_rating(user_id,business_id)
        except:
            user_rating_shift = 0
        if user_rating_shift == None:
            user_rating_shift=0
        pred_ratings.append(user_rating_shift + np.average(city_train[city_train.user_id == user_id].stars_review.values))
    return pred_ratings

def calc_base_rating(sampled_test):
    base_ratings = []
    for id_ in sampled_test.index:
        user_id = sampled_test[sampled_test.index==id_].user_id.values[0]
        base_ratings.append(np.average(city_train[city_train.user_id == user_id].stars_review.values))
    return base_ratings
#**************************************************************************
print ("Time1: ",date.datetime.now())
final_test = get_test(city_train,city_test)
#final_test.shape = (3769, 10) for Charlotte

sampled_test = {}
#10 samples of size 80% of test set
for i in range(10):        
    #sampled_test[i] = resample(final_test,n_samples=int(np.ceil(0.8 * final_test.shape[0])))
    sampled_test[i] = resample(final_test,n_samples = 100)
    sampled_test[i] = sampled_test[i].reset_index()
    sampled_test[i] = sampled_test[i].drop('index', axis = 1)

print ("Time2: ",date.datetime.now())

mean_avg_error = []
#Calculate mean average error for 10 samples
for i in range(10):
    #Predict ratings
    pred_ratings = calc_rating(sampled_test[i])
    sampled_test[i]['pred_ratings'] = pred_ratings
    sampled_test[i]['pred_ratings'][sampled_test[i]['pred_ratings']>5] = 5
    #Calculate mean average error of predicted ratings
    mean_avg_error.append(np.average(np.abs((sampled_test[i]['pred_ratings']
               -sampled_test[i]['stars_review'])/sampled_test[i]['stars_review'])))
print("Mean Average Error for 10 samples:",mean_avg_error)
#Mean Average Error for 10 samples: [0.44326004207725789, 0.50765041305984671]

print ("Time3: ",date.datetime.now())

base_mean_avg_error = []
#Calculate Base mean average error for 10 samples
for i in range(10):      
    #Calculate Base Ratings
    base_ratings = calc_base_rating(sampled_test[i])
    sampled_test[i]['base_ratings'] = base_ratings
    sampled_test[i]['base_ratings'][sampled_test[i]['base_ratings']>5] = 5
    #Calculate mean average error of base ratings
    base_mean_avg_error.append(np.average(np.abs((sampled_test[i]['base_ratings']
                           -sampled_test[i]['stars_review'])/sampled_test[i]['stars_review'])))
print("Base Mean Average Error for 10 samples:",base_mean_avg_error)
#Base Mean Average Error for 10 samples: [0.43042405334103401, 0.54015879100393527]
    
# =============================================================================

"""vectorizer_rest_rev_train = TfidfVectorizer(binary=False
                             , stop_words = 'english'
                             , min_df = 5, max_df=.8)

vectorizer_rest_rev_train = vectorizer_rest_rev_train.fit(user_reviews_train.text.apply(lambda a:a.lower()))

tfidf = vectorizer_rest_rev_train.transform(user_reviews_train.text.apply(lambda a:a.lower()))
#tfidf.shape = ((3603, 15550))

final_test = get_test(city_train,city_test)
sampled_test=final_test

pred_ratings = []

user_id = sampled_test[sampled_test.index==0].user_id.values[0]
#'9IRuYmy5YmhtNQ6ei1p-uQ'
business_id = sampled_test[sampled_test.index==0].business_id.values[0]
#'-5XuRAfrjEiMN77J4gMQZQ'
try:
    ###########################
    test_rest_user_id = city_train[city_train.business_id == business_id].copy()
    #24 X 10    
    user_reviews_train = get_user_rev_train(city_train)
    #[3603 rows x 2 columns]
  
    #target_user_vector = nmf[user_reviews_train[user_reviews_train.user_id == user_id].index[0]].copy()
    #target_user_vector = vector[user_reviews_train[user_reviews_train.user_id == user_id].index[0]].copy()
    target_user_vector = tfidf[user_reviews_train[user_reviews_train.user_id == user_id].index[0]].copy()
    
    dist_user = []
    for user_id in test_rest_user_id.user_id.values:
        dist_user.append(cosine_similarity(tfidf[user_reviews_train[user_reviews_train.user_id==user_id].index[0]],
                                           target_user_vector)[0][0])

    test_rest_user_id['similarity'] = dist_user
    
    #Filtering technique = Calc percentile and filter percentiles > 80%
    test_rest_user_id = test_rest_user_id[test_rest_user_id.rank(pct=True).similarity > .8]
     
    rating_diff = []
    for user_id in test_rest_user_id.user_id.values:
        rating_diff.append(test_rest_user_id.stars_review[test_rest_user_id.user_id==user_id].values[0]
                            -np.average(city_train[city_train.user_id ==user_id].stars_review.values))
    test_rest_user_id['rating_diff'] = rating_diff    
    ###########################
    user_rating_shift = np.average(test_rest_user_id.rating_diff)
except:
    user_rating_shift = 0
if user_rating_shift == None:
    user_rating_shift=0
pred_ratings.append(user_rating_shift + np.average(city_train[city_train.user_id == user_id].stars_review.values))
#Done"""
