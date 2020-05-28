#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import numpy as np
import pandas as pd
import re
import string
import nltk

"""Stop Words: 
A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)"""

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
print(stop)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
from sklearn import preprocessing

"""Encode target labels with value between 0 and n_classes-1.
This transformer should be used to encode target values, i.e. y, 
and not the input X."""
encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

"""In information retrieval, tf–idf or TFIDF, short for 
term frequency–inverse document frequency, is a numerical 
statistic that is intended to reflect how important a word
is to a document in a collection or corpus"""
tfid = TfidfVectorizer()
vect = CountVectorizer()    #a list of words & accompanying counts
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

responses = pd.read_csv("C:\\Users\\kimso\\Desktop\\Distress_Resume_Data\\Describe a time when you have acted as a resource for someone else.csv", usecols= ['response_id','class','response_text'], encoding='latin-1')
resumes = pd.read_csv("C:\\Users\\kimso\\Desktop\\Distress_Resume_Data\\Data Sci Resume.csv", encoding="latin-1") 


"""
responses contains user responses, to the question: 
'Describe a time where you have acted as a resource for someone else'.

If the response is classified as 'flagged', they will be alerted for help,
else, they can continue chatting
"""
responses.head()


# In[2]:


#Data Exploration

#Proportion of Flagged cases?
responses['class'].value_counts()

#What could be the keywords that induce comments to be flagged?
def cloud(text):
    wordcloud = WordCloud(background_color="white", 
                         stopwords = stopwords).generate(" ".join([i for i in text.str.upper()]))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('responses wordcloud')
cloud(responses['response_text'])


# In[3]:


#DummyCode Y Variable
responses['Label'] = encode.fit_transform(responses['class'])
responses['Label'].value_counts()


# In[56]:


# Naive Bayes Classification, assuming independence between features
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

x = responses.response_text
y = responses.Label
#default split: 0.25
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)

"""
When you do the fit method of a class, you are fitting your data 
to the specific instance, i.e., like 'training' the instance to your data. 

This could be like creating a list of words if you are doing NLP.

The transform step is then to actually apply that fit model to your data. 
E.g., populating a matrix with the counts of words for the list of words
you created in your fit step
"""

x_train_dtm = vect.fit_transform(x_train)

x_test_dtm = vect.transform(x_test)


nb.fit(x_train_dtm, y_train)

y_pred= nb.predict(x_test_dtm)

y_predict = nb.predict(x_test_dtm)
Nb_Accuracy = metrics.accuracy_score(y_test, y_predict)
print(f'Accuracy of NB Model is {Nb_Accuracy}')


# In[59]:


#let's try random forest classifier

rf = RandomForestClassifier(max_depth=10, max_features=10)
rf.fit(x_train_dtm, y_train)
rf_predict = rf.predict(x_test_dtm)
Rf_Accuracy = metrics.accuracy_score(y_test, rf_predict)
print(f'Accuracy of RF Model is {Rf_Accuracy}')


# In[61]:


chatbot_text = responses['response_text']
len(chatbot_text)


# In[62]:


tf_idf = CountVectorizer(max_features=256).fit_transform(chatbot_text.values)


# In[ ]:




