
################################################################################################################3
##################
#Task2: Using Word2vec vectors to understand LDA topics and quickly tag review to user requested topic
#Author: Shubhankar Mitra
##################
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize
from tqdm import tqdm, tqdm_pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor,FastNPExtractor
import pickle
import re
from multiprocessing import Pool
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models.word2vec import Word2Vec 
LabeledSentence = gensim.models.doc2vec.LabeledSentence 
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk.tokenize import TweetTokenizer 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error,classification_report
from sklearn.metrics import roc_auc_score
from scipy import sparse
tokenizer = TweetTokenizer()

#Data Creation
checkin=pd.read_csv('checkin.csv')

tip=pd.read_csv('tip.csv')

photos=pd.read_csv('photos.csv')

user=pd.read_csv('user.csv')

business=pd.read_csv('business.csv')

review=pd.read_csv('review.csv')



business_pitts = business[business.city == 'Pittsburgh']

business_pitts.groupby('business_id').count()

business_pitts.set_index('business_id').index.get_duplicates()

data = business_pitts.merge(review, on = 'business_id')

data_bus_review_user=data.merge(user, on='user_id')

data_bus_review_user[['business_id','name_x','user_id','name_y','stars_x'
                      ,'review_count_x','categories','stars_y','date','text'
                      ,'useful_x','funny_x','cool_x','friends']].to_csv('pittsburg.csv')
#read pittsburg data
dt = pd.read_csv('pittsburg.csv')
#Filter for restaurants
dt_rest=dt[dt.categories.apply(lambda a: 'Restaurants' in a)]
#Filter for reveiws with greater than 50 words
rest = dt_rest[dt_rest.text.apply(lambda a: len(a.split()))>50]
#Filter for 1200 restaurants
tmp=np.unique(rest.business_id)[0:1200]
rest1 = rest[rest.business_id.isin(tmp)].copy()
#Lemmatize, remove punctuations and symbols
lmtzr = WordNetLemmatizer()
sent_pre1 = rest1['text'].apply(lambda a: [lmtzr.lemmatize(word) for word in tokenizer.tokenize(a.lower())])
sent_pre2 = sent_pre1.apply(lambda a:[word for word in a if ((re.search(r'[^\w\s]', word) is None)|(len(word)>1))] )
st_W=set([lmtzr.lemmatize(word) for word in stopwords.words('english')])
tokenised_sentences =  sent_pre2  

#Create Vocabulory
vocab = []
for sent in tokenised_sentences:
    for word in sent:
        vocab.append(word)
#Remove stopwords for LDA
vocab =set(vocab)-st_W

#Creating Bag of Words
vectorizer = CountVectorizer(binary=False
                             , vocabulary = list(vocab)
                             , min_df = 5, max_df=.8)
vectorizer = vectorizer.fit(tokenised_sentences.apply\
                                                          (lambda a: " ".join(a)))

BoW=vectorizer.transform(tokenised_sentences.apply\
                                                          (lambda a: " ".join(a)))
#Set number of LDA components
n_components = 25
#Number of words for showing LDA results
n_top_words = 20

#Helper function showing LDA topic words
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

#Run LDA on bag of words of review text
lda = LatentDirichletAllocation(n_components=n_components, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(BoW)
BoW_lda = lda.transform(BoW)
#Printing LDA topic words[Optional]
tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

#Function to labelize for Word2Vec
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized



#Labelize tweet for Word2Vec training
wtv_train = labelizeTweets(tokenised_sentences, '')


#Train Word2Vec
tweet_w2v = Word2Vec(min_count=1, size=100, window=5, workers=7)
tweet_w2v.build_vocab([x.words for x in tqdm(wtv_train)])
tweet_w2v.train([x.words for x in tqdm(wtv_train)],epochs=100, total_examples=tweet_w2v.corpus_count)
#Function to calculate weighted average of word2vec vector for each topic
#Weighted based on topic importance for word
def topic_wtv(wtv_model, n_top_words, topic_model):
    ret_val = []
    for topic in topic_model.components_:
        ret_val.append(np.sum(np.array([wtv_model[item[0]]*item[1] for item in  [(tf_feature_names[i],topic[i]) for i \
                               in np.argsort(topic)[:-n_top_words - 1:-1]]]),axis=0)/np.sum(np.sort(topic)[:-n_top_words - 1:-1]))
    return np.array(ret_val)
#Word for which reviews are requested
category='entertainment'
#Column name for True column
category_label = 'show'
#Weighted average word2vec vector
topic_av_wtv = topic_wtv(tweet_w2v,lda.components_.shape[1],lda)
#Calculate similarity of word2vec vector of word with LDA topic
topic_sim = cosine_similarity(topic_av_wtv,tweet_w2v[category].reshape(1, -1))
#Taking only top 3 important topic
topic_sim[np.argsort(topic_sim.reshape((1,-1)))[0][::-1][3:]]=0
#Checking for review similarity with LDA topic distribution of review
rewiew_category = cosine_similarity(topic_sim.reshape((1,-1)), BoW_lda)
rest1['cat_sim_unscale'] = rewiew_category[0]
#Scaling review importance between 0 and 1
rest1['cat_sim'] = (rest1.cat_sim_unscale-min(rest1.cat_sim_unscale))/(max(rest1.cat_sim_unscale)-min(rest1.cat_sim_unscale))
#Creating column for ground truth
rest1['Pizza_Italian'] = rest1.categories.apply(lambda a: 1 if (\
     ('Pizza' in a)\
     |('Italian' in a)) else 0)
rest1['beverage'] = rest1.categories.apply(lambda a: 1 if (\
     ('Tea Rooms' in a)|\
     ('Wineries' in a)|\
     ('Wine Bars' in a)|\
     ('Wine & Spirits' in a)|\
     ('Pubs' in a)|\
     ('Juice Bars & Smoothies' in a)|\
     ('Bars' in a)|\
     ('Coffee & Tea' in a)|\
     ('Beer' in a)|\
     ('Beer Bar' in a)|\
     ('Bubble Tea' in a)|\
     ('Cafes' in a)|\
     ('Cocktail Bars' in a)|\
     ('Coffee & Tea' in a)|\
     ('Gastropubs' in a)) else 0)

rest1['beer'] = rest1.categories.apply(lambda a: 1 if (\
     ('Wineries' in a)|\
     ('Wine Bars' in a)|\
     ('Wine & Spirits' in a)|\
     ('Pubs' in a)|\
     ('Bars' in a)|\
     ('Beer' in a)|\
     ('Beer Bar' in a)|\
     ('Cocktail Bars' in a)|\
     ('Gastropubs' in a)) else 0)

rest1['sweet'] = rest1.categories.apply(lambda a: 1 if (\
     ('Waffles' in a)|\
     ('Ice Cream & Frozen Yogurt' in a)|\
     ('Gelato' in a)|\
     ('Desserts' in a)|\
     ('Creperies' in a)) else 0)

rest1['show'] = rest1.categories.apply(lambda a: 1 if (\
     ('Wedding Planning' in a)|\
     ('Venues & Event Spaces' in a)|\
     ('Event Planning & Services' in a)|\
     ('Arts & Entertainment' in a)) else 0)

#Evaluating with different metrics for LDA with word2vec
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
print(category)
print(average_precision_score(rest1[category_label],rest1['cat_sim']))
print(log_loss(rest1[category_label],rest1['cat_sim']))
threshold = .9999
print("Accuracy: ",accuracy_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Recall: ",recall_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Precision: ",precision_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Count of positive class: ",np.sum(rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Count of actual positive: ",np.sum(rest1[category_label]))
print(roc_auc_score(rest1[category_label],rest1['cat_sim']))


#For comparison we only use LDA to retrieve results
word_lda_topic = np.array(lda.components_.T[tf_feature_names.index(category)])
#Taking only top 3 important topic
word_lda_topic[np.argsort(word_lda_topic.reshape((1,-1)))[0][::-1][3:]]=0
#Checking for review similarity with LDA topic distribution of review
rewiew_category = cosine_similarity(word_lda_topic.reshape((1,-1)), BoW_lda)
rest1['cat_sim_unscale'] = rewiew_category[0]
#Scaling review importance between 0 and 1
rest1['cat_sim'] = (rest1.cat_sim_unscale-min(rest1.cat_sim_unscale))/(max(rest1.cat_sim_unscale)-min(rest1.cat_sim_unscale))
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss

#Evaluating with different metrics by just using LDA to retrive reviews
print(average_precision_score(rest1[category_label],rest1['cat_sim']))
print(log_loss(rest1[category_label],rest1['cat_sim']))
print("Accuracy: ",accuracy_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Recall: ",recall_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Precision: ",precision_score(rest1[category_label],rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Count of positive class: ",np.sum(rest1['cat_sim'].apply(lambda a:1 if a>threshold else 0)))
print("Count of actual positive: ",np.sum(rest1[category_label]))
print(roc_auc_score(rest1[category_label],rest1['cat_sim']))

#Checking accuracy of a simple naive bayes supervised learning method
clf = BernoulliNB()
clf.fit(BoW, rest1['beverage'])
clf.score(BoW, rest1['beverage'])
print(average_precision_score(rest1['beverage'],pd.Series(clf.predict_proba(BoW).T[1])))
print(log_loss(rest1['beverage'],pd.Series(clf.predict_proba(BoW).T[1])))

clf = BernoulliNB()
clf.fit(BoW, rest1['Pizza_Italian'])
clf.score(BoW, rest1['Pizza_Italian'])
print(average_precision_score(rest1['Pizza_Italian'],pd.Series(clf.predict_proba(BoW).T[1])))
print(log_loss(rest1['Pizza_Italian'],pd.Series(clf.predict_proba(BoW).T[1])))

#Checking a few review text
rest1.sort_values('cat_sim')[::-1].text.iloc[0:10].values
